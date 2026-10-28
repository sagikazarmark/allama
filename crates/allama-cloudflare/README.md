# allama-cloudflare

[![crates.io](https://img.shields.io/crates/v/allama-cloudflare?style=flat-square)](https://crates.io/crates/allama-cloudflare)
[![docs.rs](https://img.shields.io/docsrs/allama-cloudflare?style=flat-square)](https://docs.rs/allama-cloudflare)

Extraction models for `allama` on Cloudflare Workers AI, over its REST
API: Clairic (`@cf/cloudflare/clairic`).

allama is not affiliated with Cloudflare; Workers AI is one of the
backends it supports.

**What an extraction model is.** An extraction model does not write text.
You send it one input (an email, an invoice, a CSV of comparable sales)
and named, typed fields: a text span, a number, a date, a duration, a
list, an estimate inside bounds you declare, or a forecast of a series.
It answers every field in one pass with a value of that type, or null,
with the probability that the input does not determine it, and for text
with where in the input the value came from. In `allama` it is an
`ExtractionProvider`, a sibling of chat, embeddings and decisions. It is
**unstable** (ADR 0018): one provider exists, so the trait and its types
may change in a minor release until a second one does.

**Where the rest of Workers AI is.** Chat and embeddings on Workers AI are
`allama-openai-compat`'s `Cloudflare` preset, decisions (Clef)
`allama-systemone`'s. Clairic speaks neither of those protocols, so it has
this crate. Inside a Worker, `allama-cloudflare-worker` runs the same
model through the AI binding with this crate's codec. All of them report
the provider name `cloudflare`.

**Getting it running.** Set `CLOUDFLARE_ACCOUNT_ID` and
`CLOUDFLARE_API_TOKEN` (a token with Workers AI access), then
`Provider::from_env()` with the model `@cf/cloudflare/clairic`
(`cloudflare:@cf/cloudflare/clairic` through `allama-providers`).

```bash
cargo run -p allama-examples --example provider_extraction
```

```rust
use allama::extraction::{Field, ListItemKind};
use allama::{ExtractionProvider, ExtractionRequest};
use allama_cloudflare::Provider;

# async fn example() -> allama::Result<()> {
let provider = Provider::from_env()?;
let email = "Order A-88412 arrived on Sept 28 with a cracked screen and a dented corner. \
             I paid $649.00 plus $52.10 tax.";
let request = ExtractionRequest::new("@cf/cloudflare/clairic", email)
    .field("order_id", Field::text().instructions("Order number"))
    .field("delivered_on", Field::date().instructions("Delivery date"))
    .field(
        "amount_paid",
        Field::number().decimals(2).unit("USD").instructions("Total paid including tax"),
    )
    .field("damage", Field::list(ListItemKind::Text).instructions("Each kind of damage described"))
    .field("tracking_number", Field::text());

let response = provider.extract_values(&request).await?;
let order_id = response.text("order_id").unwrap();
println!("{:?} at {:?}", order_id.value, order_id.span.map(|s| s.text(&request.input)));
println!("paid {:?}", response.number("amount_paid").unwrap().value); // Some(701.1), computed
println!("damage {:?}", response.list("damage").unwrap().texts().collect::<Vec<_>>());
println!("tracking null? {}", response.text("tracking_number").unwrap().is_null());
# Ok(()) }
```

## Mapping

| allama | Clairic |
| --- | --- |
| `ExtractionRequest::model` | the URL (`.../ai/run/{model}`) and the body's `model` selector, the name's last segment (`clairic`) |
| `input` | `state` |
| `fields` | `fields`, in order; every parameter as given: `instructions`, `integer`, `decimals`, `unit`, `items`, `min`, `max`, `step`, `scale`, `horizon`, the period as `interval` |
| `images` (inline data) | `images` as `{"content_type", "base64"}`; a URL or provider file is refused, an image detail warned about |
| `reference_date` | `reference_date` |
| `TextAnswer::span`, `ListItem::span` | the model's character offsets, placed as byte spans by offset conversion or by finding the value; unplaceable spans are dropped with a `dropped.span` warning |
| `null_probability` | `p_null` |
| `EstimateAnswer::{value, median, interval}` | `value`, `median`, `p10`–`p90` as an 80% interval |
| `ForecastAnswer::steps` | parallel arrays (`value[]`, `median[]`, `p10[]`, `p90[]`) or `steps[]`, whichever the model sends |
| `Usage` | `input_tokens`, `output_tokens`; `state_tokens`, `image_tokens`, `schema_tokens` as the breakdowns `text_input_tokens`, `image_input_tokens`, `schema_input_tokens` |
| provider data `Confidence`, `Guesses` | `confidence`, `guess` per field |
| provider data `Distributions` | `mean`, `p_below`, `p_above`, `quantiles` per estimate or forecast step |
| provider data `RunDetails` | `passes`, `truncated` (also a warning coded `approximated.input_truncated`), `state_text`, the token counts as reported |
| provider data `AnswerExtras` | keys of an answer this adapter does not model (`source`, …) |
| `ExtractionRequestOptions` | per-field `null_threshold`; `options.include_quantiles`, `options.return_state_text`; `raw_response` keeps the body as `RawResponse`; `extra` for fields the adapter does not model |

The provider answers `Supported` for every field kind and `Unknown` for
`ImageInput` (whether a model reads images is model data, ADR 0015), and
claims the limits Clairic's input schema fixes as `ExtractionLimits`: at
most 4 images, a forecast horizon of at most 64 (`ProviderBuilder::limits`
replaces them). The rest of the schema is checked before sending (field
names of at most 64 of `[A-Za-z0-9_.-]` starting with a letter or `_`,
`decimals` up to 6, images of 4 MiB each and 8 MiB together), an
`Error::InvalidRequest` that names the field. Options and provider data
use the namespace `cloudflare-workers-ai`, shared with
`allama-cloudflare-worker`.

## Testing

`tests/replay.rs` replays the documentation's example response and the
documented error codes (`fixtures/`), `tests/conformance.rs` runs every
extraction contract of `allama-conformance`, and `tests/e2e.rs` checks
the live endpoint (`--ignored`, needs the two variables).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
