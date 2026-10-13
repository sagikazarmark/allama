//! Every `Supported` embedding capability claim must be backed by evidence.
//! Chat claims are probed by the capability honesty contract
//! (`src/conformance_tests.rs`).

use std::path::PathBuf;

use allama::EmbeddingCapability as E;
use allama_conformance::FixtureDir;
use allama_conformance::capabilities::{Evidence, assert_embedding_capabilities_backed};
use allama_gemini::Provider;

#[test]
fn embedding_claims_are_backed() {
    let provider = Provider::builder()
        .api_key("test-key")
        .http_transport(allama_http::ReplayTransport::default())
        .build()
        .unwrap();
    let fixtures = FixtureDir::new(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures"));
    assert_embedding_capabilities_backed(
        &provider,
        "gemini-embedding-001",
        &fixtures,
        &[
            (E::BatchInput, Evidence::E2e("batch_embed")),
            (E::OutputDimensions, Evidence::E2e("dimensions")),
            (
                E::InputType,
                Evidence::Fixture("embed_input_type_query_wire.json"),
            ),
        ],
    );
    assert_embedding_capabilities_backed(
        &provider,
        "gemini-embedding-2",
        &fixtures,
        &[
            (E::BatchInput, Evidence::E2e("batch_embed")),
            (E::OutputDimensions, Evidence::E2e("dimensions")),
            (E::ImageInput, Evidence::E2e("image_embed")),
            (E::CompositeInput, Evidence::E2e("composite_embed")),
        ],
    );
}
