//! What an extraction plan reports through Anthropic, for the typical
//! schemas in each mode, without sending anything.

use allama_anthropic::Provider;
use allama_http::ReplayTransport;

#[test]
fn plans() {
    let transport = ReplayTransport::new([]);
    let provider = Provider::builder()
        .api_key("k")
        .http_transport(transport.clone())
        .build()
        .unwrap();
    // Anthropic removes `minimum`, so an unsigned integer is approximate;
    // `output_config.format` has no unconstrained form, so a map is refused
    // in native mode (an `Auto` extraction falls back to forced-tool mode).
    let expected = [
        "Native Person: exact",
        "Native Order: approximate",
        "Native Tagged: refused response_format",
        "ForcedTool Person: exact",
        "ForcedTool Order: approximate",
        "ForcedTool Tagged: not_enforced",
    ];
    assert_eq!(
        typical::plans(&provider, "claude-haiku-4-5-20251001"),
        expected
    );
    assert!(transport.requests().is_empty());
}

// The fixture types are only planned for, never built.
#[allow(dead_code)]
mod typical {
    use std::collections::HashMap;

    use allama::chat::extract::{ChatExtractionMode, ChatExtractor};
    use allama::schemars::JsonSchema;
    use allama::{ChatProvider, ChatRequest, ValidationCoverage};
    use serde::Deserialize;

    /// Strings, one of them optional.
    #[derive(Deserialize, JsonSchema)]
    #[schemars(crate = "allama::schemars")]
    pub struct Person {
        name: String,
        nickname: Option<String>,
    }

    /// An unsigned integer (schemars writes `minimum: 0`).
    #[derive(Deserialize, JsonSchema)]
    #[schemars(crate = "allama::schemars")]
    pub struct Order {
        id: String,
        quantity: u32,
    }

    /// A map.
    #[derive(Deserialize, JsonSchema)]
    #[schemars(crate = "allama::schemars")]
    pub struct Tagged {
        title: String,
        labels: HashMap<String, String>,
    }

    /// The plan of each typical schema in each explicit mode, as
    /// `"<mode> <type>: <enforcement>"`, or `refused <field>` when the plan
    /// fails before sending. Every plan is a dry run (`Encoding`).
    pub fn plans<P: ChatProvider + ?Sized>(provider: &P, model: &str) -> Vec<String> {
        let mut rows = Vec::new();
        for mode in [ChatExtractionMode::Native, ChatExtractionMode::ForcedTool] {
            let request = ChatRequest::new(model).user("x").with_option(mode);
            let extractor = ChatExtractor::new();
            for (name, plan) in [
                ("Person", extractor.plan::<Person>(provider, &request)),
                ("Order", extractor.plan::<Order>(provider, &request)),
                ("Tagged", extractor.plan::<Tagged>(provider, &request)),
            ] {
                let cell = match plan {
                    Ok(plan) => {
                        assert_eq!(plan.coverage, ValidationCoverage::Encoding, "{name}");
                        plan.enforcement.to_string()
                    }
                    Err(err) => format!(
                        "refused {}",
                        err.unsupported_field()
                            .map_or("-".to_owned(), |f| f.to_string())
                    ),
                };
                rows.push(format!("{mode:?} {name}: {cell}"));
            }
        }
        rows
    }
}
