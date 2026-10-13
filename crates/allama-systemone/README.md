# allama-systemone

[![crates.io](https://img.shields.io/crates/v/allama-systemone?style=flat-square)](https://crates.io/crates/allama-systemone)
[![docs.rs](https://img.shields.io/docsrs/allama-systemone?style=flat-square)](https://docs.rs/allama-systemone)

Decision models for `allama` over the System One protocol
(`POST /v1/systemone`): TypeSafe's hosted API (Jev), Ollama's local
endpoint (Nimble, Tev1), gateways that forward it, and Cloudflare Workers
AI's Clef models.

allama is not affiliated with [Ollama](https://ollama.com); Ollama is one of
the backends it supports.

**What a decision model is.** A decision model does not write text. You
send it one input (a ticket, a message, a JSON record) and a few typed
questions about it: yes or no, which of these options, or how high on
this scale. It answers each with probabilities, such as "billing 0.97,
bug 0.03", in one fast call. That suits classification, routing and
moderation, where a chat model would need a prompt and output parsing.
In `allama` it is a `DecisionProvider`, a sibling of chat.

**The names.** System One is the protocol: TypeSafe defined
`POST /v1/systemone`, and Ollama and several gateways serve the same
format. TypeSafe is the company with the hosted API; its model is Jev
(`jev-latest`). Nimble and Tev1 are decision models Ollama runs locally.
Clef (`@cf/cloudflare/clef`, `@cf/cloudflare/clef-flash`) is Cloudflare's
multimodal decision model on Workers AI, which speaks System One with
extensions of its own, images among them.

**Getting one running.** Either backend takes a few minutes:

- **Locally, with Ollama 0.35 or later**, no key: `ollama pull nimble`,
  then use the `Ollama` preset with the model `nimble` (or
  `ollama:nimble` through `allama-providers`). An older Ollama has no
  System One endpoint and fails with `Error::Unsupported`, which names
  the version.
- **Hosted, with TypeSafe**: set `TYPESAFE_API_KEY` to a key from
  TypeSafe, then use the `TypeSafe` preset with `jev-latest`
  (`typesafe:jev-latest`).
- **Hosted, with Cloudflare**: set `CLOUDFLARE_ACCOUNT_ID` and
  `CLOUDFLARE_API_TOKEN` (a token with Workers AI access), then use the
  `Cloudflare` preset with `@cf/cloudflare/clef`
  (`cloudflare:@cf/cloudflare/clef`). Inside a Worker,
  `allama-cloudflare-worker` runs the same models through the AI binding.

```bash
# TypeSafe when TYPESAFE_API_KEY is set, Cloudflare when CLOUDFLARE_API_TOKEN is, else Ollama.
cargo run -p allama-examples --example provider_decision
# With a screenshot, for a model that reads images (Clef).
cargo run -p allama-examples --example provider_decision -- "Checkout fails." error.png
```

Through a LiteLLM proxy, the `LiteLlm` preset calls the proxy's TypeSafe
pass-through, `/typesafe/v1/systemone`. The proxy adds the TypeSafe key, so
it needs `TYPESAFE_API_KEY` in its own environment and a release that has
the pass-through; a proxy without it answers `404`, which fails with
`Error::Unsupported` and says so.

**One input per request.** A request carries one input and any number of
questions about it (up to the preset's limits). To classify a batch, send
one request per input, concurrently, in your own code, for example with
`futures::stream::iter(..).buffer_unordered(8)`.

`Provider` implements `allama::DecisionProvider`; `providers` has presets:


| Preset | Endpoint | Credential | Limits |
| --- | --- | --- | --- |
| `TypeSafe` | `https://api.typesafe.ai/v1/systemone` (`TYPESAFE_BASE_URL`) | `TYPESAFE_API_KEY` | 255 options, 10 levels |
| `Ollama` | `http://127.0.0.1:11434/v1/systemone` (`OLLAMA_HOST`, read as Ollama reads it) | none | 64 questions, 26 options, 26 levels |
| `OpenRouter` | `https://openrouter.ai/api/v1/systemone` | `OPENROUTER_API_KEY` | TypeSafe's |
| `DigitalOcean` | `https://inference.do-ai.run/v1/systemone` | `MODEL_ACCESS_KEY` | TypeSafe's |
| `LiteLlm` | `{proxy}/typesafe/v1/systemone` (`LITELLM_PROXY_URL`) | `LITELLM_API_KEY` | TypeSafe's |
| `Cloudflare` | `https://api.cloudflare.com/client/v4/accounts/{id}/ai/run/{model}` | `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN` | none: each model's own |

The typed example below uses `allama`'s `derive` feature
(`allama = { version = "0.1", features = ["derive"] }`) and `anyhow = "1"`.

```rust,no_run
use allama::decision::{Choice, ChoiceSet, Decision, YesNoAnswer};
use allama::DecisionProviderExt;
use allama_systemone::providers::Ollama;

#[derive(Debug, ChoiceSet)]
#[decision(rename_all = "lowercase")]
enum Label {
    #[decision(description = "Payments and refunds")]
    Billing,
    #[decision(description = "Software errors")]
    Bug,
}

#[derive(Decision)]
struct Triage {
    #[decision(instructions = "Which label fits this ticket?")]
    label: Choice<Label>,
    #[decision(instructions = "Is the customer requesting a refund?")]
    refund: YesNoAnswer,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let decided = Ollama::from_env()?
        .decide_as::<Triage>("nimble", "I was charged twice. Please refund me.")
        .await?;
    let triage = decided.value;
    println!("{:?}, refund p = {:.2}", triage.label.value(), triage.refund.probability);
    Ok(())
}
```

## Mapping

| Portable | System One |
| --- | --- |
| `DecisionRequest::input` (text or JSON) | `state` |
| `Question::yes_no` (+ `yes`/`no` descriptions) | `noul` (+ `criteria: {"true", "false"}`) |
| `Question::choice` options | `choice` with `criteria: {key: description \| null}` |
| `Question::score` levels | `score` with `criteria: ["label: description" \| "label", ...]` |
| `YesNoAnswer::probability` | `noul` |
| `ChoiceAnswer::chosen` | `choice` |
| `ScoreAnswer::probabilities` | `probabilities: {"0": .., "1": ..}` (the `legend` echo is dropped) |
| provider data `Confidence` | `confidence` (formulas differ between TypeSafe and Ollama) |
| `DecisionResponse::id`; provider data `Gateway` | OpenRouter's `id`; its `provider` and `usage.cost` |
| `DecisionRequest::images` (inline data) | `images`: bare base64 on Ollama, `{"content_type", "base64"}` on Cloudflare; refused before sending on TypeSafe and the gateways, and as a URL or provider file everywhere |
| `DecisionRequestOptions::keep_alive` | Ollama's `keep_alive` (the Ollama preset only: elsewhere it is dropped with a warning) |

### Cloudflare (Clef)

The `Cloudflare` preset posts each request to the model's own URL
(`.../ai/run/@cf/cloudflare/clef`) and sends Clef's model selector, the
last segment of the name (`clef`), as the body's `model`. The output comes
in Cloudflare's `{"result": ..}` envelope and errors in its
`{"errors": [{"code", "message"}]}` one; Workers AI's codes refine the
status (`3036` is `QuotaExhausted`, `3040` `Overloaded`, `5007` and `7000`
`ModelNotFound`).

Workers AI runs any model at that endpoint, each with its own input
schema, so the preset checks nothing a schema decides: no limits, image
types or question-name rules (ADR 0015). Clef's schema allows 64
questions, 255 options, 10 levels, 4 PNG, JPEG or WebP images and question
names of at most 100 of `[A-Za-z0-9_.-]`; it refuses anything else with
HTTP 422, an `Error::InvalidRequest` whose upstream message names the
field. To check them before sending, set `ProviderBuilder::limits` on the
preset's builder. The provider name is
`cloudflare`, as on `allama-openai-compat`'s Cloudflare preset, so the
`cloudflare` registry key offers chat, embeddings and decisions on one
account.

```rust,no_run
use allama::decision::Question;
use allama::{DecisionProvider, DecisionRequest, Image};
use allama_systemone::providers::Cloudflare;

async fn is_outage(ticket: &str, screenshot: Vec<u8>) -> allama::Result<f64> {
    let provider = Cloudflare::from_env()?;
    let request = DecisionRequest::new("@cf/cloudflare/clef", ticket)
        .image(Image::bytes("image/png", screenshot))
        .question("outage", Question::yes_no("Is this a service outage?"));
    let response = provider.decide(&request).await?;
    Ok(response.yes_no("outage").map_or(0.0, |answer| answer.probability))
}
```

Requests are checked against the preset's limits before anything is sent.
Responses are checked against the request: every question answered by its
kind, choices among the options, and exactly one probability per option or
level. `DecisionRequestOptions::extra` adds body fields but cannot replace
the ones the request sets.

Errors map from each server's envelope: Ollama's `{"error"}`, TypeSafe's
`{"detail"}`, and the gateways' `{"error": {..}}`. Two cases fail with
`Error::Unsupported`:

- a model that cannot serve System One, for example a chat model on Ollama
  ("does not support decision");
- a server without the endpoint, for example Ollama before 0.35.

The live tests (`tests/e2e_ollama.rs`, `tests/e2e_typesafe.rs`,
`tests/e2e_cloudflare.rs`) run with `--ignored`; the Ollama ones skip on a
server older than 0.35. The scheduled e2e workflow runs the Cloudflare ones
next to the compat crate's Cloudflare tests; no scheduled job runs the
TypeSafe and Ollama ones yet. The replay tests
cover the wire format from recorded and documented examples
(`fixtures/README.md`).

Input beyond the model's context fails with `Error::ContextLengthExceeded`.

## Features

- `reqwest` (default): the default transport. With default features off,
  build the provider with `http_transport(...)` (any `allama_http::HttpTransport`);
  the crate then builds without `reqwest`, and the constructors that would
  build a `reqwest` client (`new`, `from_env`) are not compiled.
- `default-tls` (default), `rustls`, `native-tls`, `rustls-ring`: the TLS
  backend of the default `reqwest` transport (no effect on `wasm32`, where
  the host's `fetch` does TLS); each implies `reqwest`. `rustls-ring` uses
  `ring` instead of `aws-lc-rs`, so static musl builds need no cmake. With
  `reqwest` on and default features off, pick one or HTTPS endpoints cannot
  be reached.
- `http-tracing`: implies `reqwest` and turns on `allama-http`'s
  `http-tracing`, which wraps every client a `ReqwestTransport` builds (or
  gets through `ReqwestTransport::from_client(..)`) with `reqwest-tracing`,
  so HTTP spans fire on outgoing requests. Every adapter shares one
  `allama-http`, so enabling it on any provider crate traces the HTTP
  requests of every provider in the build, not only this one's.

To send through your own `reqwest-middleware` stack, enable `allama-http`'s
`reqwest-middleware` feature and pass
`http_transport(ReqwestTransport::from_middleware(..))`.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
