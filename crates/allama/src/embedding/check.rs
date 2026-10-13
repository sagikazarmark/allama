use crate::{EmbeddingRequest, EmbeddingResponse, Error, Result};

/// Check that `response` holds vectors a caller can use for `request`.
///
/// An adapter calls it on every response it decodes, before
/// [`Prepared::finish`](crate::adapter::embedding::Prepared::finish), and attributes
/// the error to itself ([`Error::with_provider`]). It fails with
/// [`Error::UnexpectedResponse`] when:
///
/// - the response does not hold one vector per input;
/// - a vector is empty;
/// - [`dimensions`](EmbeddingRequest::dimensions) was requested and a vector
///   has another size;
/// - the vectors have different sizes;
/// - a value is not finite (NaN or infinite, as a value beyond the range of
///   `f32` decodes).
///
/// A vector index built from such vectors is silently wrong, so the
/// response is never repaired. The message names the model and both sizes:
///
/// ```
/// use allama::adapter::embedding::check_response;
/// use allama::{EmbeddingRequest, EmbeddingResponse, Error};
///
/// let request = EmbeddingRequest::new("nomic-embed-text").input("a").dimensions(256);
/// let response = EmbeddingResponse::new(vec![vec![0.1; 768]]);
/// let err = check_response(&request, &response).unwrap_err();
/// assert!(matches!(err, Error::UnexpectedResponse { .. }));
/// assert_eq!(
///     err.message(),
///     "requested 256 dimensions, but `nomic-embed-text` returned 768 (input 0)"
/// );
/// ```
///
/// # Errors
///
/// [`Error::UnexpectedResponse`] for the first problem found.
pub fn check_response(request: &EmbeddingRequest, response: &EmbeddingResponse) -> Result<()> {
    let (model, inputs, vectors) = (
        request.model.as_str(),
        request.inputs.len(),
        response.embeddings.len(),
    );
    if vectors != inputs {
        return Err(unexpected(format!(
            "`{model}` returned {vectors} vectors for {inputs} inputs"
        )));
    }
    check_vectors(model, request.dimensions, None, 0, &response.embeddings).map(|_| ())
}

/// Check the shape and values of `vectors`, the vectors of the inputs from
/// `first_input` on, against the `requested` dimensions and, when given,
/// the size `expected` of the vectors before them. Returns the size of
/// these vectors (`expected` when there are none).
pub(crate) fn check_vectors(
    model: &str,
    requested: Option<u32>,
    expected: Option<usize>,
    first_input: usize,
    vectors: &[Vec<f32>],
) -> Result<Option<usize>> {
    let mut size = expected;
    for (index, vector) in vectors.iter().enumerate() {
        let input = first_input + index;
        let len = vector.len();
        if len == 0 {
            return Err(unexpected(format!(
                "`{model}` returned an empty vector (input {input})"
            )));
        }
        if let Some(requested) = requested
            && usize::try_from(requested).ok() != Some(len)
        {
            return Err(unexpected(format!(
                "requested {requested} dimensions, but `{model}` returned {len} (input {input})"
            )));
        }
        match size {
            Some(size) if size != len => {
                return Err(unexpected(format!(
                    "`{model}` returned vectors of different sizes: {size} dimensions before \
                     input {input}, {len} for input {input}"
                )));
            }
            Some(_) => {}
            None => size = Some(len),
        }
        if let Some(value) = vector.iter().find(|value| !value.is_finite()) {
            return Err(unexpected(format!(
                "`{model}` returned a non-finite value ({value}) (input {input})"
            )));
        }
    }
    Ok(size)
}

fn unexpected(message: String) -> Error {
    Error::unexpected_response(message).authored()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(inputs: usize) -> EmbeddingRequest {
        EmbeddingRequest::new("m").inputs((0..inputs).map(|i| i.to_string()))
    }

    fn check(request: &EmbeddingRequest, vectors: Vec<Vec<f32>>) -> Result<()> {
        check_response(request, &EmbeddingResponse::new(vectors))
    }

    #[test]
    fn accepts_vectors_of_one_size() {
        check(&request(2), vec![vec![0.1, 0.2], vec![0.3, 0.4]]).unwrap();
        check(&request(1).dimensions(3), vec![vec![0.0, -1.0, 1.0]]).unwrap();
    }

    #[test]
    fn rejects_unusable_vectors_naming_the_model_and_both_sizes() {
        for (request, vectors, message) in [
            (
                request(2),
                vec![vec![0.1]],
                "`m` returned 1 vectors for 2 inputs",
            ),
            (
                request(2).dimensions(256),
                vec![vec![0.1; 256], vec![0.1; 768]],
                "requested 256 dimensions, but `m` returned 768 (input 1)",
            ),
            (
                request(3),
                vec![vec![0.1, 0.2], vec![0.1, 0.2], vec![0.1]],
                "`m` returned vectors of different sizes: 2 dimensions before input 2, 1 for \
                 input 2",
            ),
            (
                request(2),
                vec![vec![], vec![]],
                "`m` returned an empty vector (input 0)",
            ),
            (
                request(1),
                vec![vec![0.1, f32::NAN]],
                "`m` returned a non-finite value (NaN) (input 0)",
            ),
            (
                request(1),
                vec![vec![f32::INFINITY]],
                "`m` returned a non-finite value (inf) (input 0)",
            ),
        ] {
            let err = check(&request, vectors).unwrap_err();
            assert!(matches!(err, Error::UnexpectedResponse { .. }), "{err}");
            assert_eq!(err.message(), message);
            assert_eq!(err.message_origin(), crate::MessageOrigin::Authored);
        }
    }

    #[test]
    fn later_vectors_are_checked_against_an_expected_size() {
        assert_eq!(
            check_vectors("m", None, None, 0, &[vec![0.1, 0.2]]).unwrap(),
            Some(2)
        );
        assert_eq!(check_vectors("m", None, Some(2), 4, &[]).unwrap(), Some(2));
        let err = check_vectors("m", None, Some(2), 4, &[vec![0.1, 0.2, 0.3]]).unwrap_err();
        assert_eq!(
            err.message(),
            "`m` returned vectors of different sizes: 2 dimensions before input 4, 3 for input 4"
        );
    }

    /// A JSON number beyond the range of `f32` decodes as infinity, which is
    /// how a non-finite value reaches an adapter.
    #[test]
    fn an_out_of_range_number_decodes_as_infinity() {
        let vector: Vec<f32> = serde_json::from_str("[0.5, 1e39]").unwrap();
        assert!(vector[1].is_infinite());
        let err = check(&request(1), vec![vector]).unwrap_err();
        assert!(matches!(err, Error::UnexpectedResponse { .. }), "{err}");
    }
}
