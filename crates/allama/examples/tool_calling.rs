//! A tool round trip with a schema generated from a Rust type.
//!
//! The tool is strict. Strict modes that require every property (OpenAI's)
//! make an optional property without a schema `default` nullable, so the
//! model may send `null` for `units`; `parse_arguments_with` removes it
//! before deserializing, and the field gets its default.

use allama::mock::MockProvider;
use allama::prelude::*;
use allama::schemars::{self, JsonSchema};
use serde::Deserialize;
use serde_json::json;

/// Arguments of the weather tool.
#[derive(Debug, Deserialize, JsonSchema)]
struct WeatherArgs {
    /// City name to look up.
    city: String,
    /// Temperature units; Celsius when left out.
    #[serde(default)]
    units: Units,
}

/// Temperature units. Not `Serialize`, so the schema carries no `default`.
#[derive(Debug, Default, Deserialize, JsonSchema)]
enum Units {
    #[default]
    Celsius,
    Fahrenheit,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let provider = MockProvider::tool_round_trip(
        "call_weather_1",
        "lookup_weather",
        // What a strict-mode model sends for the optional `units`.
        json!({ "city": "San Francisco", "units": null }),
        "The weather in San Francisco is cool and foggy.",
    );

    let weather = ToolDefinition::from_schema::<WeatherArgs>("lookup_weather")
        .description("Look up the weather for a city.")
        .strict(true);
    let mut request = ChatRequest::new("demo-model")
        .system("Use the weather tool when a city lookup is needed.")
        .user("What is the weather in San Francisco?")
        .tools([weather.clone()])
        .tool_choice(ToolChoice::Required);

    let first = provider.chat(&request).await?;
    request.push_message(first.to_assistant_message());
    for call in first.tool_calls() {
        let args: WeatherArgs = call.parse_arguments_with(&weather)?;
        println!(
            "tool call {} -> {} ({:?})",
            call.name, args.city, args.units
        );
        request.push_tool_result(call, format!("Weather for {}: cool and foggy.", args.city));
    }
    // `Required` forces a tool call on every request that carries it: reset
    // it, or the follow-up must call a tool again instead of answering.
    request.tool_choice = None;

    let second = provider.chat(&request).await?;
    println!("final text: {}", second.text_or_empty());
    Ok(())
}
