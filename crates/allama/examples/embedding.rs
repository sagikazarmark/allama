//! Batch embeddings, single-text embeddings, chunked batches (resumable
//! after a failure), and the query/document input type for retrieval.

use allama::mock::MockEmbeddingProvider;
use allama::prelude::*;
use allama::{ChunkedEmbeddingProgress, EmbeddingInputType, EmbeddingResponse};
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let provider = MockEmbeddingProvider::new([
        Ok(EmbeddingResponse::new(vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
        ])),
        Ok(EmbeddingResponse::new(vec![vec![0.7, 0.8, 0.9]])),
        Ok(EmbeddingResponse::new(vec![vec![1.0], vec![2.0]])),
        Ok(EmbeddingResponse::new(vec![vec![3.0]])),
        Ok(EmbeddingResponse::new(vec![vec![0.5, 0.5]])),
        Ok(EmbeddingResponse::new(vec![vec![1.0], vec![2.0]])),
        Err(Error::rate_limited("slow down")),
        Ok(EmbeddingResponse::new(vec![vec![3.0], vec![4.0]])),
        Ok(EmbeddingResponse::new(vec![vec![5.0]])),
    ]);

    let request = EmbeddingRequest::new("demo-embedding-model")
        .inputs(["hello world", "embedding demo"])
        .dimensions(3);
    for (index, vector) in provider
        .embed(&request)
        .await?
        .embeddings
        .iter()
        .enumerate()
    {
        println!("embedding[{index}] = {vector:?}");
    }

    let single = provider
        .embed_input("demo-embedding-model", "quick one-shot")
        .await?;
    println!("single-input vector: {single:?}");

    // Split a large batch into requests of at most two inputs.
    let batch = EmbeddingRequest::new("demo-embedding-model").inputs(["a", "b", "c"]);
    let combined = provider.embed_chunked(&batch, 2).await?;
    println!("chunked vectors: {:?}", combined.embeddings);

    // Retrieval: embed documents with `Document` at index time and searches
    // with `Query`. Providers that take a hint map it (Gemini `taskType`,
    // `input_type` on some OpenAI-compatible endpoints); the others drop it
    // with a warning, or fail under `UnsupportedPolicy::Error`.
    let search = EmbeddingRequest::new("demo-embedding-model")
        .input("Which language is simple?")
        .input_type(EmbeddingInputType::Query);
    let response = provider.embed(&search).await?;
    println!(
        "query vector: {:?} ({} warnings)",
        response.embeddings[0],
        response.warnings.len()
    );

    // A long indexing job: store each chunk as it arrives, and resume from
    // the failed chunk instead of paying for the batch again.
    let corpus = EmbeddingRequest::new("demo-embedding-model").inputs(["a", "b", "c", "d", "e"]);
    let mut stored: Vec<Vec<f32>> = Vec::new();
    let mut next = 0;
    while next < corpus.inputs.len() {
        let mut rest = corpus.clone();
        rest.inputs.drain(..next);
        let mut chunks = std::pin::pin!(provider.embed_chunks(&rest, 2));
        while let Some(chunk) = chunks.next().await {
            match chunk {
                Ok(chunk) => {
                    println!(
                        "stored inputs {}..{}",
                        next + chunk.offset,
                        next + chunk.end()
                    );
                    stored.extend(chunk.response.embeddings);
                }
                Err(err) if err.is_retryable() => {
                    let progress = ChunkedEmbeddingProgress::from_error(&err)
                        .expect("a chunked call attaches its progress");
                    println!(
                        "chunk failed ({err}); resuming at input {}",
                        next + progress.completed_inputs
                    );
                }
                Err(err) => return Err(err.into()),
            }
        }
        next = stored.len();
    }
    println!("indexed {} vectors", stored.len());
    Ok(())
}
