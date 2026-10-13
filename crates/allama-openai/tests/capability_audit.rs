//! Every `Supported` embedding capability claim must be backed by evidence
//! Chat claims are probed by the capability honesty contract
//! (`conformance.rs`).

use std::path::PathBuf;

use allama::EmbeddingCapability as E;
use allama_conformance::FixtureDir;
use allama_conformance::capabilities::{Evidence, assert_embedding_capabilities_backed};
use allama_openai::Provider;

#[test]
fn embedding_claims_are_backed() {
    let provider = Provider::builder()
        .api_key("test-key")
        .http_transport(allama_http::ReplayTransport::default())
        .build()
        .unwrap();
    let fixtures = FixtureDir::new(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures"));
    for model in ["text-embedding-3-small", "text-embedding-3-large"] {
        assert_embedding_capabilities_backed(
            &provider,
            model,
            &fixtures,
            &[
                (E::BatchInput, Evidence::E2e("batch_embed")),
                (E::OutputDimensions, Evidence::E2e("dimensions")),
                (
                    E::InputType,
                    Evidence::Test("symmetric_models_take_any_input_type"),
                ),
            ],
        );
    }
    assert_embedding_capabilities_backed(
        &provider,
        "text-embedding-ada-002",
        &fixtures,
        &[
            (E::BatchInput, Evidence::E2e("batch_embed")),
            (
                E::InputType,
                Evidence::Test("symmetric_models_take_any_input_type"),
            ),
        ],
    );
}
