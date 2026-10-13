//! Grounds the neutral embedding input model (ADR 0012) in providers that
//! allama has no adapter for yet.
//!
//! These test-only encoders map an [`EmbeddingRequest`] onto the documented
//! request bodies of Voyage multimodal embeddings and Cohere Embed v4, and
//! enforce each provider's documented rules. If the neutral model could not
//! express a request these APIs accept, or had to fake a rule, these tests are
//! where it would show.
//!
//! - Voyage: <https://docs.voyageai.com/reference/multimodal-embeddings-api>
//! - Cohere: <https://docs.cohere.com/reference/embed>

use allama::{EmbeddingInput, EmbeddingPart, EmbeddingRequest, Error, MediaSource, Result};
use serde_json::{Value, json};

/// Voyage `POST /v1/multimodalembeddings`: each input is an interleaved
/// `content` list of text and images; one request uses either URL or base64
/// images, not both.
fn voyage_body(request: &EmbeddingRequest) -> Result<Value> {
    let mut url_sources = false;
    let mut inline_sources = false;
    let mut inputs = Vec::new();
    for input in &request.inputs {
        let mut content = Vec::new();
        for part in input.parts() {
            match part {
                EmbeddingPart::Text(text) => {
                    content.push(json!({"type": "text", "text": text.text}));
                }
                EmbeddingPart::Image(image) => match &image.source {
                    MediaSource::Url { url } => {
                        url_sources = true;
                        content.push(json!({"type": "image_url", "image_url": url}));
                    }
                    MediaSource::Base64 {
                        media_type, data, ..
                    } => {
                        inline_sources = true;
                        content.push(json!({
                            "type": "image_base64",
                            "image_base64": format!("data:{media_type};base64,{data}"),
                        }));
                    }
                    _ => return Err(Error::unsupported("unsupported image source")),
                },
                _ => return Err(Error::unsupported("unsupported part")),
            }
        }
        inputs.push(json!({"content": content}));
    }
    if url_sources && inline_sources {
        return Err(Error::unsupported(
            "Voyage requests use either image URLs or base64 images, not both",
        ));
    }
    Ok(json!({"inputs": inputs, "model": request.model}))
}

/// Cohere `POST /v2/embed` with `inputs`: each input's `content` mixes text
/// and images into one embedding; images are data URIs only.
fn cohere_body(request: &EmbeddingRequest, input_type: &str) -> Result<Value> {
    let inputs = request
        .inputs
        .iter()
        .map(|input| {
            let content = input
                .parts()
                .iter()
                .map(|part| match part {
                    EmbeddingPart::Text(text) => Ok(json!({"type": "text", "text": text.text})),
                    EmbeddingPart::Image(image) => match &image.source {
                        MediaSource::Base64 {
                            media_type, data, ..
                        } => Ok(json!({
                            "type": "image_url",
                            "image_url": {"url": format!("data:{media_type};base64,{data}")},
                        })),
                        _ => Err(Error::unsupported(
                            "Cohere embeds images from data URIs only",
                        )),
                    },
                    _ => Err(Error::unsupported("unsupported part")),
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(json!({"content": content}))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(json!({
        "model": request.model,
        "input_type": input_type,
        "embedding_types": ["float"],
        "inputs": inputs,
    }))
}

fn product() -> EmbeddingInput {
    EmbeddingInput::from(vec![
        EmbeddingPart::text("Red trail-running shoe"),
        EmbeddingPart::image_base64("image/png", "iVBORw0KGgo"),
    ])
}

#[test]
fn voyage_accepts_interleaved_inputs() {
    let request = EmbeddingRequest::new("voyage-multimodal-3")
        .input(product())
        .input("Red trail-running shoe");
    assert_eq!(
        voyage_body(&request).unwrap(),
        json!({
            "inputs": [
                {"content": [
                    {"type": "text", "text": "Red trail-running shoe"},
                    {"type": "image_base64", "image_base64": "data:image/png;base64,iVBORw0KGgo"}
                ]},
                {"content": [{"type": "text", "text": "Red trail-running shoe"}]}
            ],
            "model": "voyage-multimodal-3"
        })
    );
}

#[test]
fn voyage_rejects_mixed_image_sources_instead_of_converting_them() {
    let request = EmbeddingRequest::new("voyage-multimodal-3")
        .input(product())
        .input(EmbeddingInput::image_url("https://example.com/shoe.jpg"));
    let err = voyage_body(&request).unwrap_err();
    assert!(matches!(err, Error::Unsupported { .. }), "{err}");

    let urls_only = EmbeddingRequest::new("voyage-multimodal-3")
        .input(EmbeddingInput::image_url("https://example.com/shoe.jpg"));
    assert_eq!(
        voyage_body(&urls_only).unwrap()["inputs"][0]["content"][0],
        json!({"type": "image_url", "image_url": "https://example.com/shoe.jpg"})
    );
}

#[test]
fn cohere_v4_combines_text_and_image_in_one_input() {
    let request = EmbeddingRequest::new("embed-v4.0").input(product());
    assert_eq!(
        cohere_body(&request, "search_document").unwrap(),
        json!({
            "model": "embed-v4.0",
            "input_type": "search_document",
            "embedding_types": ["float"],
            "inputs": [{"content": [
                {"type": "text", "text": "Red trail-running shoe"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo"}}
            ]}]
        })
    );
}

#[test]
fn cohere_rejects_url_images_instead_of_downloading_them() {
    let request = EmbeddingRequest::new("embed-v4.0")
        .input(EmbeddingInput::image_url("https://example.com/shoe.jpg"));
    let err = cohere_body(&request, "search_document").unwrap_err();
    assert!(matches!(err, Error::Unsupported { .. }), "{err}");
}
