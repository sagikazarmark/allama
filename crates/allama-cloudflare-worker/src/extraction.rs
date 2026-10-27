//! `ExtractionProvider` implementation for Cloudflare Workers AI.
//!
//! Runs Clairic extraction models (`@cf/cloudflare/clairic`) through the AI
//! binding. Their input and output are `allama-cloudflare`'s wire format
//! without Cloudflare's REST envelope, encoded and decoded with that
//! crate's codec, so both transports send and read alike, and share the
//! options type and provider data (namespace `cloudflare-workers-ai`).
//!
//! Unstable, like [`allama::ExtractionProvider`] (ADR 0018).

use allama::adapter::extraction::Prepared;
use allama::{
    CapabilitySupport, ExtractionCapability, ExtractionProvider, ExtractionRequest,
    ExtractionResponse, ExtractionValidation, Result, ValidationCoverage,
};
use allama_cloudflare::ExtractionRequestOptions;
use allama_cloudflare::binding as codec;
use serde::Serialize;

use crate::binding::run_text;
use crate::options::report_options_under_name;
use crate::{PROVIDER_NAME, Provider};

impl ExtractionProvider for Provider {
    async fn extract_values(&self, request: &ExtractionRequest) -> Result<ExtractionResponse> {
        let options = resolve(request)?;
        let mut prepared = allama::adapter::extraction::prepare(self, request)?;
        let input = encode(&mut prepared, request, &options)?;
        prepared.mark_sent();
        let output = run_text(&self.ai, &request.model, &input, &options.run_options).await?;
        decode(prepared, &output, &options)
    }

    /// A dry run of `extract_values`: the preflight and encoding, without
    /// running the model ([`ValidationCoverage::Encoding`]).
    fn validate_extraction_request(
        &self,
        request: &ExtractionRequest,
    ) -> Result<ExtractionValidation> {
        validate(self, request)
    }

    fn extraction_capability(
        &self,
        _model: &str,
        capability: ExtractionCapability,
    ) -> CapabilitySupport {
        builtin_extraction_capability(capability)
    }
}

/// Model-independent extraction capability answers, as the HTTP provider
/// (`allama_cloudflare::Provider`) gives them: every field kind, and
/// `Unknown` for images, which are model data. Like it, the binding claims
/// no limits (ADR 0015).
fn builtin_extraction_capability(capability: ExtractionCapability) -> CapabilitySupport {
    codec::builtin_extraction_capability(capability)
}

fn resolve(request: &ExtractionRequest) -> Result<ExtractionRequestOptions> {
    request
        .resolve_option::<ExtractionRequestOptions>()
        .map_err(|err| err.with_provider(PROVIDER_NAME))
}

/// Everything before the binding runs, shared by the call and its dry run:
/// the input the binding receives.
fn encode<'a>(
    prepared: &mut Prepared<'_>,
    request: &'a ExtractionRequest,
    options: &'a ExtractionRequestOptions,
) -> Result<impl Serialize + use<'a>> {
    let d = prepared.degradations_mut();
    report_options_under_name(&request.options, "ExtractionRequestOptions", d);
    codec::encode_workers_ai(request, options, d).map_err(|err| err.with_provider(PROVIDER_NAME))
}

/// The binding's output, as JSON text, as the response to `prepared`,
/// checked against the request, with its warnings.
fn decode(
    prepared: Prepared<'_>,
    output: &str,
    options: &ExtractionRequestOptions,
) -> Result<ExtractionResponse> {
    let response = codec::decode_workers_ai(prepared.request(), output, options, PROVIDER_NAME)
        .map_err(|err| err.with_provider(PROVIDER_NAME))?;
    Ok(prepared.finish(response))
}

/// A dry run of `extract_values`.
fn validate<P: ExtractionProvider + ?Sized>(
    provider: &P,
    request: &ExtractionRequest,
) -> Result<ExtractionValidation> {
    let options = resolve(request)?;
    let mut prepared = allama::adapter::extraction::prepare_dry_run(provider, request)?;
    encode(&mut prepared, request, &options)?;
    Ok(prepared.into_validation(ValidationCoverage::Encoding))
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use allama::extraction::{Field, ListItemKind};
    use allama::{Error, Image, ProviderIdentity};
    use allama_cloudflare::{Confidence, RunDetails};
    use allama_conformance::probe::ProbeOutcome;
    use serde_json::{Map, Value, json};

    use super::*;
    use crate::RawResponse;

    /// Stands in for [`Provider`], whose binding exists only in the Workers
    /// runtime: the same capability answers, call path and validation, with
    /// the model's output canned and the inputs the binding would receive
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

    impl ExtractionProvider for Binding {
        async fn extract_values(&self, request: &ExtractionRequest) -> Result<ExtractionResponse> {
            let options = resolve(request)?;
            let mut prepared = allama::adapter::extraction::prepare(self, request)?;
            let input = encode(&mut prepared, request, &options)?;
            prepared.mark_sent();
            let input = serde_json::to_value(&input).expect("the input serializes");
            self.sent.lock().unwrap().push(input);
            let output = self
                .output
                .clone()
                .unwrap_or_else(|| answers(request).to_string());
            decode(prepared, &output, &options)
        }

        fn validate_extraction_request(
            &self,
            request: &ExtractionRequest,
        ) -> Result<ExtractionValidation> {
            validate(self, request)
        }

        fn extraction_capability(
            &self,
            _model: &str,
            capability: ExtractionCapability,
        ) -> CapabilitySupport {
            builtin_extraction_capability(capability)
        }
    }

    /// Clairic's output answering every field of `request`; text answers
    /// are the input's first word, so every span places.
    fn answers(request: &ExtractionRequest) -> Value {
        let word = request.input.split_whitespace().next();
        let text = || match word {
            Some(word) => {
                let start = request.input.find(word).unwrap_or(0);
                json!({"type": "text", "value": word, "span": [start, start + word.len()], "p_null": 0.01, "confidence": 0.9, "guess": word})
            }
            None => {
                json!({"type": "text", "value": null, "p_null": 1.0, "confidence": 0.5, "guess": "x"})
            }
        };
        let answers: Map<String, Value> = request
            .fields
            .iter()
            .map(|(name, field)| {
                let answer = match field {
                    Field::Text(_) => text(),
                    Field::Number(_) => json!({"type": "number", "value": 7, "p_null": 0.01, "confidence": 0.9, "guess": 7}),
                    Field::Date(_) => json!({"type": "date", "value": "2026-09-28", "p_null": 0.01, "confidence": 0.9}),
                    Field::Time(_) => json!({"type": "time", "value": "13:34:00", "p_null": 0.01, "confidence": 0.9}),
                    Field::DateTime(_) => json!({"type": "datetime", "value": "2026-09-28T13:34:00", "p_null": 0.01, "confidence": 0.9}),
                    Field::Duration(duration) => json!({"type": "duration", "value": 3, "unit": duration.unit.as_str(), "p_null": 0.01, "confidence": 0.9}),
                    Field::List(list) => match list.items {
                        ListItemKind::Number => json!({"type": "list", "value": [1, 2]}),
                        ListItemKind::Date => json!({"type": "list", "value": ["2026-09-28"]}),
                        _ => match word {
                            Some(word) => {
                                let start = request.input.find(word).unwrap_or(0);
                                json!({"type": "list", "value": [word], "spans": [[start, start + word.len()]]})
                            }
                            None => json!({"type": "list", "value": []}),
                        },
                    },
                    Field::Estimate(estimate) => {
                        let mid = f64::midpoint(estimate.min, estimate.max);
                        json!({"type": "estimate", "value": mid, "mean": mid, "median": mid, "p10": estimate.min, "p90": estimate.max, "p_below": 0.0, "p_above": 0.0})
                    }
                    Field::Forecast(forecast) => {
                        let mid = f64::midpoint(forecast.min, forecast.max);
                        let series: Vec<f64> = (0..forecast.horizon).map(|_| mid).collect();
                        json!({"type": "forecast", "value": series, "median": series, "p10": series, "p90": series})
                    }
                    _ => unreachable!("every field kind is answered"),
                };
                (name.clone(), answer)
            })
            .collect();
        json!({
            "model": "clairic",
            "answers": answers,
            "truncated": false,
            "usage": {"state_tokens": 20, "image_tokens": 0, "schema_tokens": 30, "input_tokens": 50, "output_tokens": 0, "passes": 1}
        })
    }

    async fn probe(request: ExtractionRequest) -> ProbeOutcome<ExtractionResponse> {
        let binding = Binding::default();
        let result = binding.extract_values(&request).await;
        ProbeOutcome::new(result, binding.sent())
    }

    fn fixture(name: &str) -> String {
        std::fs::read_to_string(format!("{}/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap()
    }

    /// The example of Cloudflare's Clairic page, answered with its documented
    /// output (`fixtures/extraction_invoice.json`, the `result` of
    /// `allama-cloudflare`'s `fixtures/clairic/invoice.json`).
    #[tokio::test]
    async fn extracts_with_the_documented_output() {
        let request = ExtractionRequest::new(
            "@cf/cloudflare/clairic",
            "Invoice INV-2291 from Northwind Traders, issued 2026-09-02, is due 2026-10-02. Total: $1,080.00, net 30 days. Contacts: Dana Ruiz, Priya Shah.",
        )
        .field("invoice_number", Field::text().instructions("Invoice number"))
        .field("due_date", Field::date())
        .field("total", Field::number().decimals(2).unit("USD"))
        .field("contacts", Field::list(ListItemKind::Text))
        .field("po_number", Field::text())
        .field("days_to_pay", Field::estimate(0.0, 120.0))
        .with_option(
            ExtractionRequestOptions::new()
                .raw_response()
                .run_option("gateway", json!({"id": "my-gateway"})),
        );
        let binding = Binding::answering(fixture("extraction_invoice.json"));
        let response = binding.extract_values(&request).await.unwrap();

        let sent = binding.sent();
        assert_eq!(sent[0]["model"], "clairic");
        assert_eq!(sent[0]["fields"]["total"]["decimals"], 2);
        assert_eq!(response.provider.as_deref(), Some(PROVIDER_NAME));
        let invoice_number = response.text("invoice_number").unwrap();
        assert_eq!(invoice_number.value.as_deref(), Some("INV-2291"));
        assert_eq!(
            invoice_number.span.unwrap().text(&request.input),
            Some("INV-2291")
        );
        assert!(response.text("po_number").unwrap().is_null());
        assert_eq!(response.estimate("days_to_pay").unwrap().median, Some(30.0));
        assert_eq!(response.usage.as_ref().unwrap().input_tokens, Some(655));
        let confidence = response.provider_data.get::<Confidence>().unwrap().unwrap();
        assert_eq!(confidence.0["po_number"], 0.9995);
        let run = response.provider_data.get::<RunDetails>().unwrap().unwrap();
        assert_eq!(run.passes, Some(3));
        // Both crates' `RawResponse` read the one entry.
        let raw = response
            .provider_data
            .get::<RawResponse>()
            .unwrap()
            .unwrap();
        assert_eq!(raw.0["model"], "clairic");
        assert!(response.warnings.is_empty(), "{:?}", response.warnings);
    }

    #[tokio::test]
    async fn images_and_extra_fields_reach_the_input() {
        let request = ExtractionRequest::new("@cf/cloudflare/clairic", "A receipt.")
            .image(Image::base64("image/webp", "aGk="))
            .field("total", Field::number())
            .with_option(ExtractionRequestOptions::new().extra("seed", json!(7)));
        let binding = Binding::default();
        binding.extract_values(&request).await.unwrap();
        let sent = &binding.sent()[0];
        assert_eq!(
            sent["images"],
            json!([{"content_type": "image/webp", "base64": "aGk="}])
        );
        assert_eq!(sent["seed"], 7);

        for owned in ["model", "state", "fields", "images"] {
            let request = ExtractionRequest::new("@cf/cloudflare/clairic", "x")
                .field("f", Field::text())
                .with_option(ExtractionRequestOptions::new().extra(owned, json!(1)));
            let err = binding.extract_values(&request).await.unwrap_err();
            assert!(
                matches!(err, Error::InvalidRequest { .. }),
                "{owned}: {err:?}"
            );
        }
    }

    /// Options under the provider name are not read, and say so.
    #[tokio::test]
    async fn options_under_the_provider_name_are_reported() {
        let request = ExtractionRequest::new("@cf/cloudflare/clairic", "x")
            .field("f", Field::text())
            .with_options_json(PROVIDER_NAME, json!({"raw_response": true}));
        let response = Binding::default().extract_values(&request).await.unwrap();
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
            ExtractionRequest::new("@cf/cloudflare/clairic", "x").field("f", Field::text());
        for output in [
            r#"{"answers":{}}"#,
            "null",
            r#"{"answers":{"f":{"type":"number","value":1}}}"#,
        ] {
            let err = Binding::answering(output)
                .extract_values(&request)
                .await
                .unwrap_err();
            assert!(
                matches!(err, Error::UnexpectedResponse { .. }),
                "{output}: {err:?}"
            );
        }
    }

    /// Every extraction contract, through the binding stand-in. The binding
    /// has no HTTP response or headers, so the error mapping and transport
    /// contracts do not apply.
    #[tokio::test]
    async fn the_adapter_passes_every_contract() {
        use allama_conformance::{Contract, ExtractionAdapterUnderTest, assert_extraction_adapter};

        let under_test = ExtractionAdapterUnderTest::new(["@cf/cloudflare/clairic"], probe).skip([
            Contract::ErrorKindsMapped,
            Contract::ErrorLogExcludesBody,
            Contract::RetrySignalsSurfaced,
            Contract::TransportOptionsApplied,
        ]);
        assert_extraction_adapter(&Binding::default(), &under_test).await;
    }
}
