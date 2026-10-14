//! The sampling drift check: `temperature(0.2)` sent to each vendor's
//! newest model is answered, or refused as the provider's own refusal of a
//! field, never as an invalid request.
//!
//! Adapters send sampling as asked and map a model's refusal to
//! `Error::Unsupported { stage: Provider, subject: Field }` (ADR 0015). The
//! patterns behind that mapping are pinned by recorded fixtures; this check
//! is what notices when a new model words its refusal differently, which
//! would degrade it to `InvalidRequest` and stop fallback from failing over.
//!
//! Ignored by default. The weekly `e2e.yaml` job (`newest`, one matrix
//! entry per vendor) resolves the newest model id from each vendor's models
//! API with `scripts/newest-model.sh` and passes it in:
//!
//! ```sh
//! ANTHROPIC_NEWEST_MODEL=$(scripts/newest-model.sh anthropic) \
//!     cargo test -p allama-examples --test newest_models anthropic -- --ignored
//! GEMINI_NEWEST_MODEL=$(scripts/newest-model.sh gemini) \
//!     cargo test -p allama-examples --test newest_models gemini -- --ignored
//! OPENAI_NEWEST_MODEL=$(scripts/newest-model.sh openai) \
//!     cargo test -p allama-examples --test newest_models openai -- --ignored
//! ```
//!
//! The script reads the vendor's key from the environment or `.env`; the
//! test needs it in the environment.

use allama::{ChatProvider, ChatRequest, Error, Subject, UnsupportedStage};

fn newest_model(variable: &str) -> String {
    std::env::var(variable).unwrap_or_else(|_| {
        panic!("{variable} must name the vendor's newest model (the weekly e2e job resolves it)")
    })
}

/// One short call with `temperature(0.2)`: `Ok`, or `Unsupported` at stage
/// `Provider` naming a field. Anything else, an `InvalidRequest` above all,
/// means a refusal the adapter no longer recognizes.
async fn assert_sampling_is_answered_or_refused(
    endpoint: &str,
    provider: &impl ChatProvider,
    model: &str,
) {
    let request = ChatRequest::new(model)
        .user("Say ok.")
        .max_output_tokens(256)
        .temperature(0.2);
    let label = format!("{endpoint}:{model}");
    match provider.chat(&request).await {
        Ok(response) => println!("{label}: answered, warnings {:?}", response.warnings),
        Err(err) => {
            assert!(
                matches!(err, Error::Unsupported { .. })
                    && err.unsupported_stage() == Some(UnsupportedStage::Provider)
                    && matches!(err.unsupported_subject(), Some(Subject::Field { .. })),
                "{label}: expected Ok or Unsupported {{ stage: Provider, subject: Field }}, got \
                 {}: {err}",
                err.kind_name()
            );
            println!("{label}: refused {:?}", err.unsupported_field());
        }
    }
}

#[tokio::test]
#[ignore]
async fn anthropic_newest_model_sampling() {
    let provider = allama_anthropic::Provider::from_env().expect("ANTHROPIC_API_KEY must be set");
    assert_sampling_is_answered_or_refused(
        "anthropic",
        &provider,
        &newest_model("ANTHROPIC_NEWEST_MODEL"),
    )
    .await;
}

#[tokio::test]
#[ignore]
async fn gemini_newest_model_sampling() {
    let provider = allama_gemini::Provider::from_env().expect("GEMINI_API_KEY must be set");
    assert_sampling_is_answered_or_refused(
        "gemini",
        &provider,
        &newest_model("GEMINI_NEWEST_MODEL"),
    )
    .await;
}

/// Both OpenAI endpoints: Chat Completions and Responses word their
/// refusals differently.
#[tokio::test]
#[ignore]
async fn openai_newest_model_sampling() {
    let model = newest_model("OPENAI_NEWEST_MODEL");
    let chat = allama_openai::Provider::from_env().expect("OPENAI_API_KEY must be set");
    assert_sampling_is_answered_or_refused("openai-chat", &chat, &model).await;
    let responses =
        allama_openai::ResponsesProvider::from_env().expect("OPENAI_API_KEY must be set");
    assert_sampling_is_answered_or_refused("openai", &responses, &model).await;
}
