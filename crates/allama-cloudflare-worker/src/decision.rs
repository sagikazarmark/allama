//! `DecisionProvider` implementation for Cloudflare Workers AI.
//!
//! Runs Clef decision models (`@cf/cloudflare/clef`,
//! `@cf/cloudflare/clef-flash`) through the AI binding. Their input and
//! output are the System One protocol with Cloudflare's extensions, encoded
//! and decoded with `allama-systemone`'s codec, the one its `Cloudflare`
//! preset uses over HTTP, so both transports send and read alike.

use allama::adapter::decision::Prepared;
use allama::{
    CapabilitySupport, DecisionCapability, DecisionProvider, DecisionRequest, DecisionResponse,
    DecisionValidation, Result, ValidationCoverage,
};
use allama_systemone::binding as codec;
use serde::Serialize;

use crate::binding::run_text;
use crate::options::report_options_under_name;
use crate::{DecisionRequestOptions, PROVIDER_NAME, Provider, RawResponse};

impl DecisionProvider for Provider {
    async fn decide(&self, request: &DecisionRequest) -> Result<DecisionResponse> {
        let options = resolve(request)?;
        let mut prepared = allama::adapter::decision::prepare(self, request)?;
        let input = encode(&mut prepared, request, &options)?;
        prepared.mark_sent();
        let output = run_text(&self.ai, &request.model, &input, &options.run_options).await?;
        decode(prepared, &output, options.raw_response)
    }

    /// A dry run of `decide`: the preflight and encoding, without running
    /// the model ([`ValidationCoverage::Encoding`]).
    fn validate_decision_request(&self, request: &DecisionRequest) -> Result<DecisionValidation> {
        validate(self, request)
    }

    fn decision_capability(
        &self,
        _model: &str,
        capability: DecisionCapability,
    ) -> CapabilitySupport {
        builtin_decision_capability(capability)
    }
}

/// Model-independent decision capability answers, as the HTTP preset
/// (`allama_systemone::providers::Cloudflare`) gives them: every question
/// kind; images are model data. Like the preset, the binding claims no
/// limits: each model's input schema is its own (ADR 0015).
fn builtin_decision_capability(capability: DecisionCapability) -> CapabilitySupport {
    match capability {
        DecisionCapability::YesNo | DecisionCapability::Choice | DecisionCapability::Score => {
            CapabilitySupport::Supported
        }
        _ => CapabilitySupport::Unknown,
    }
}

fn resolve(request: &DecisionRequest) -> Result<DecisionRequestOptions> {
    request
        .resolve_option::<DecisionRequestOptions>()
        .map_err(|err| err.with_provider(PROVIDER_NAME))
}

/// Everything before the binding runs, shared by the call and its dry run:
/// the input the binding receives.
fn encode<'a>(
    prepared: &mut Prepared<'_>,
    request: &'a DecisionRequest,
    options: &'a DecisionRequestOptions,
) -> Result<impl Serialize + use<'a>> {
    let d = prepared.degradations_mut();
    report_options_under_name(&request.options, "DecisionRequestOptions", d);
    codec::encode_workers_ai(request, &options.extra, d)
        .map_err(|err| err.with_provider(PROVIDER_NAME))
}

/// The binding's output, as JSON text, as the response to `prepared`,
/// checked against the request, with its warnings.
fn decode(prepared: Prepared<'_>, output: &str, raw_response: bool) -> Result<DecisionResponse> {
    let mut response = codec::decode_workers_ai(prepared.request(), output, PROVIDER_NAME)?;
    if raw_response {
        let raw: serde_json::Value = serde_json::from_str(output)?;
        response
            .provider_data
            .insert(&RawResponse(raw))
            .map_err(|err| err.with_provider(PROVIDER_NAME))?;
    }
    Ok(prepared.finish(response))
}

/// A dry run of `decide`.
fn validate<P: DecisionProvider + ?Sized>(
    provider: &P,
    request: &DecisionRequest,
) -> Result<DecisionValidation> {
    let options = resolve(request)?;
    let mut prepared = allama::adapter::decision::prepare_dry_run(provider, request)?;
    encode(&mut prepared, request, &options)?;
    Ok(prepared.into_validation(ValidationCoverage::Encoding))
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use allama::decision::Question;
    use allama::{Error, Image, ProviderIdentity};
    use allama_conformance::probe::ProbeOutcome;
    use serde_json::{Map, Value, json};

    use super::*;

    /// Stands in for [`Provider`], whose binding exists only in the Workers
    /// runtime: the same capability answers, call path and validation, with
    /// the model's output canned (an answer to every question unless
    /// `output` says otherwise) and the inputs the binding would receive
    /// captured.
    #[derive(Default)]
    struct Binding {
        sent: std::sync::Mutex<Vec<Value>>,
        output: Option<String>,
    }

    impl Binding {
        fn answering(output: impl Into<String>) -> Self {
            Self {
                output: Some(output.into()),
                ..Self::default()
            }
        }

        fn sent(&self) -> Vec<Value> {
            self.sent.lock().unwrap().clone()
        }
    }

    impl ProviderIdentity for Binding {
        fn provider_name(&self) -> &str {
            PROVIDER_NAME
        }

        fn reads_namespace(&self, namespace: &str) -> bool {
            namespace == crate::NAMESPACE
        }
    }

    impl DecisionProvider for Binding {
        async fn decide(&self, request: &DecisionRequest) -> Result<DecisionResponse> {
            let options = resolve(request)?;
            let mut prepared = allama::adapter::decision::prepare(self, request)?;
            let input = encode(&mut prepared, request, &options)?;
            prepared.mark_sent();
            let input = serde_json::to_value(&input).expect("the input serializes");
            self.sent.lock().unwrap().push(input);
            let output = self
                .output
                .clone()
                .unwrap_or_else(|| answers(request).to_string());
            decode(prepared, &output, options.raw_response)
        }

        fn validate_decision_request(
            &self,
            request: &DecisionRequest,
        ) -> Result<DecisionValidation> {
            validate(self, request)
        }

        fn decision_capability(
            &self,
            _model: &str,
            capability: DecisionCapability,
        ) -> CapabilitySupport {
            builtin_decision_capability(capability)
        }
    }

    /// Clef's output answering every question of `request`.
    fn answers(request: &DecisionRequest) -> Value {
        let answers: Map<String, Value> = request
            .questions
            .iter()
            .map(|(name, question)| {
                let answer = match question {
                    Question::Choice(choice) => {
                        let share = 1.0 / choice.options.len() as f64;
                        let probabilities: Map<String, Value> = choice
                            .options
                            .iter()
                            .map(|option| (option.key.clone(), json!(share)))
                            .collect();
                        json!({"type": "choice", "choice": choice.options[0].key, "probabilities": probabilities, "confidence": 0.0})
                    }
                    Question::Score(score) => {
                        let probabilities: Map<String, Value> = (0..score.levels.len())
                            .map(|level| (level.to_string(), json!(if level == 0 { 1.0 } else { 0.0 })))
                            .collect();
                        json!({"type": "score", "score": 0.0, "legend": {}, "probabilities": probabilities, "confidence": 1.0})
                    }
                    _ => json!({"type": "noul", "noul": 0.75}),
                };
                (name.clone(), answer)
            })
            .collect();
        json!({"model": "clef", "answers": answers, "usage": {"input_tokens": 1, "output_tokens": 0}})
    }

    async fn probe(request: DecisionRequest) -> ProbeOutcome<DecisionResponse> {
        let binding = Binding::default();
        let result = binding.decide(&request).await;
        ProbeOutcome::new(result, binding.sent())
    }

    fn fixture(name: &str) -> String {
        std::fs::read_to_string(format!("{}/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap()
    }

    /// The example of Cloudflare's Clef page, answered with the output
    /// recorded over HTTP (`fixtures/decision_triage.json`, the `result`
    /// of `allama-systemone`'s `fixtures/cloudflare/triage.json`).
    #[tokio::test]
    async fn decides_with_the_recorded_output() {
        let request = DecisionRequest::new(
            "@cf/cloudflare/clef",
            "Checkout has been failing for every customer for the last hour.",
        )
        .question(
            "urgent",
            Question::yes_no("Is this support request urgent?"),
        )
        .question(
            "team",
            Question::choice("Which team should handle this request?")
                .option("billing", "Payments, invoices, and refunds")
                .option("technical", "Outages, errors, and configuration")
                .option("sales", "Plans and upgrades"),
        )
        .question(
            "severity",
            Question::score("How severe is the customer impact?").levels([
                "No impact",
                "Minor",
                "Major",
                "Critical",
            ]),
        )
        .with_option(DecisionRequestOptions::new().raw_response(true));
        let binding = Binding::answering(fixture("decision_triage.json"));
        let response = binding.decide(&request).await.unwrap();

        let sent = binding.sent();
        assert_eq!(sent[0]["model"], "clef");
        assert_eq!(
            sent[0]["questions"]["severity"]["criteria"],
            json!(["No impact", "Minor", "Major", "Critical"])
        );
        assert_eq!(response.provider.as_deref(), Some(PROVIDER_NAME));
        assert_eq!(response.choice("team").unwrap().chosen, "technical");
        assert_eq!(response.yes_no("urgent").unwrap().probability, 0.9895);
        assert_eq!(response.usage.unwrap().input_tokens, Some(346));
        let confidence = response
            .provider_data
            .get::<crate::Confidence>()
            .unwrap()
            .unwrap();
        assert_eq!(confidence.0["severity"], 0.9012);
        let raw = response
            .provider_data
            .get::<RawResponse>()
            .unwrap()
            .unwrap();
        assert_eq!(raw.0["model"], "clef");
    }

    #[tokio::test]
    async fn images_and_extra_fields_reach_the_input() {
        let request = DecisionRequest::new("@cf/cloudflare/clef", "A product photo.")
            .image(Image::base64("image/webp", "aGk="))
            .question("q", Question::yes_no("Is it damaged?"))
            .with_option(DecisionRequestOptions::new().extra("seed", 7));
        let binding = Binding::default();
        binding.decide(&request).await.unwrap();
        let sent = &binding.sent()[0];
        assert_eq!(
            sent["images"],
            json!([{"content_type": "image/webp", "base64": "aGk="}])
        );
        assert_eq!(sent["seed"], 7);

        for owned in ["model", "STATE", "questions", "images"] {
            let request = DecisionRequest::new("@cf/cloudflare/clef", "x")
                .question("q", Question::yes_no("?"))
                .with_option(DecisionRequestOptions::new().extra(owned, 1));
            let err = binding.decide(&request).await.unwrap_err();
            assert!(
                matches!(err, Error::InvalidRequest { .. }),
                "{owned}: {err:?}"
            );
        }
    }

    /// Options under the provider name are not read, and say so.
    #[tokio::test]
    async fn options_under_the_provider_name_are_reported() {
        let request = DecisionRequest::new("@cf/cloudflare/clef", "x")
            .question("q", Question::yes_no("?"))
            .with_options_json(PROVIDER_NAME, json!({"raw_response": true}));
        let response = Binding::default().decide(&request).await.unwrap();
        assert_eq!(response.warnings.len(), 1, "{:?}", response.warnings);
        assert!(
            response
                .provider_data
                .get::<RawResponse>()
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn output_that_answers_another_request_is_refused() {
        let request =
            DecisionRequest::new("@cf/cloudflare/clef", "x").question("q", Question::yes_no("?"));
        for output in [
            r#"{"answers":{}}"#,
            "null",
            r#"{"answers":{"q":{"type":"choice"}}}"#,
        ] {
            let err = Binding::answering(output)
                .decide(&request)
                .await
                .unwrap_err();
            assert!(
                matches!(err, Error::UnexpectedResponse { .. }),
                "{output}: {err:?}"
            );
        }
    }

    /// Every decision contract, through the binding stand-in. The binding
    /// has no HTTP response or headers, so the error mapping and transport
    /// contracts do not apply.
    #[tokio::test]
    async fn the_adapter_passes_every_contract() {
        use allama_conformance::{Contract, DecisionAdapterUnderTest, assert_decision_adapter};

        let under_test = DecisionAdapterUnderTest::new(
            ["@cf/cloudflare/clef", "@cf/cloudflare/clef-flash"],
            probe,
        )
        .skip([
            Contract::ErrorKindsMapped,
            Contract::ErrorLogExcludesBody,
            Contract::RetrySignalsSurfaced,
            Contract::TransportOptionsApplied,
        ]);
        assert_decision_adapter(&Binding::default(), &under_test).await;
    }
}
