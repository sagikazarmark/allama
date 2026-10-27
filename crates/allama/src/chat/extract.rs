//! Structured extraction into typed Rust values.
//!
//! # Typical use
//!
//! Derive `Deserialize` and `JsonSchema` for the type, extract it from any
//! chat provider, and tell the failures apart by their reason:
//!
//! ```
//! # #[cfg(feature = "mock")]
//! # tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async {
//! # let provider = allama::mock::MockProvider::with_text(r#"{"author": "Ada", "year": 1843}"#);
//! use allama::chat::extract::{ChatExtractError, ChatExtractExt};
//! use allama::schemars::{self, JsonSchema};
//! use allama::{ChatRequest, Error};
//!
//! #[derive(serde::Deserialize, JsonSchema)]
//! struct Program {
//!     author: String,
//!     year: u16,
//! }
//!
//! let request = ChatRequest::new("gpt-5-mini").user("Ada published the first program in 1843.");
//! match provider.extract::<Program>(&request).await {
//!     Ok(extracted) => println!("{} ({})", extracted.value.author, extracted.value.year),
//!     Err(Error::Extract { failure, .. }) => match failure.error() {
//!         ChatExtractError::Refused { .. } => println!("the model declined"),
//!         ChatExtractError::Truncated { .. } => println!("cut off: raise max_output_tokens or lower the reasoning effort"),
//!         other => println!("unusable output: {other}"),
//!     },
//!     // A provider error (a rate limit, a timeout) is returned as itself.
//!     Err(error) => return Err(error),
//! }
//! # Ok::<(), Error>(()) }).unwrap();
//! ```
//!
//! To extract from a chat that used tools, see
//! [extracting from a conversation](ChatExtractor#extracting-from-a-conversation):
//! a request with tools fails with [`ChatExtractError::ToolConflict`].
//!
//! To test code that extracts, script the value the model returns with
//! `MockResponse::structured` (feature `mock`): it answers in the form the
//! extraction asks for, whichever mode it picks.
//!
//! The [`ChatExtractFailure`] also keeps the output that could not be used
//! ([`response`](ChatExtractFailure::response)) and what every pass cost
//! ([`metadata`](ChatExtractFailure::metadata)); see
//! [Bookkeeping](#bookkeeping).
//!
//! An extraction derives its passes from the [`ChatRequest`] it is given
//! and keeps every field (system prompts, messages, sampling settings,
//! provider options) except the ones the selected mode replaces: the
//! response format in `Native` mode, the tools and tool choice in
//! `ForcedTool` mode. A request that carries tools of its own fails with
//! [`ChatExtractError::ToolConflict`] in **every** mode, before anything is
//! sent: the model would be offered both your tools and the schema, and
//! might call a tool instead of answering, so the same request must not
//! work on one provider and fail on another. To extract from a
//! conversation that used tools, start a fresh request from its messages
//! or clear the tools on a copy; see
//! [extracting from a conversation](ChatExtractor#extracting-from-a-conversation).
//!
//! # Contents
//!
//! - [Choosing a mode](#choosing-a-mode)
//! - [Automatic fallback to forced-tool mode](#automatic-fallback-to-forced-tool-mode)
//! - [Recovery is opt-in](#recovery-is-opt-in)
//! - [Bookkeeping](#bookkeeping)
//! - [Output that is not an answer](#output-that-is-not-an-answer)
//! - [Reasoning models](#reasoning-models)
//! - [Schema enforcement](#schema-enforcement)
//! - [Enforcement floor and plans](#enforcement-floor-and-plans)
//! - [Undeclared keys](#undeclared-keys)
//! - [Target types](#target-types)
//! - [Validating values](#validating-values)
//!
//! # Choosing a mode
//!
//! Extraction derives a provider request that constrains the output to the
//! JSON Schema of `T`, sends it, and parses the result:
//!
//! - [`ChatExtractionMode::Native`] uses the provider's JSON Schema response
//!   format;
//! - [`ChatExtractionMode::ForcedTool`] forces a call to a synthetic tool whose
//!   arguments follow the schema;
//! - [`ChatExtractionMode::Auto`] (default) picks from the provider's capability
//!   answers, a verified mode before a guessed one: `Native` when structured
//!   output is `Supported`; else `ForcedTool` when tool calls and forced tool
//!   choice are both `Supported`; else `Native` when structured output is
//!   `Unknown`; else `ForcedTool` when neither tool answer is `Unsupported`.
//!   When the answers rule out both, it fails with
//!   [`ChatExtractError::Unsupported`]: set the mode, or settle the answers with
//!   a capability resolver
//!   ([`CapabilityOverrideProvider`](crate::wrappers::CapabilityOverrideProvider),
//!   or the adapter's own `with_chat_capabilities`).
//!
//! # Automatic fallback to forced-tool mode
//!
//! Reporting structured-output support does not mean every schema fits the
//! provider's native output. When `Auto` picked `Native` and the adapter
//! refuses the response format before sending (an [`Error::Unsupported`]
//! whose [`unsupported_field`](Error::unsupported_field) is
//! [`RequestField::ResponseFormat`] and whose
//! [stage](Error::unsupported_stage) is
//! [`BeforeSending`](crate::UnsupportedStage::BeforeSending), such as Anthropic's
//! `output_config.format` refusing a map), the extraction retries the pass
//! in `ForcedTool` mode. On [`ChatExtractExt::extract_stream`] the refusal may
//! come from the `chat_stream` call or as the stream's first item, before
//! any event; both switch modes. To see that item, `extract_stream` reads
//! the first item before returning whenever `Auto` picked `Native`; the
//! returned stream still yields it.
//!
//! The switch is recorded, not hidden:
//! [`ChatExtractionMetadata::mode`] reports `ForcedTool` and
//! [`ChatExtractionMetadata::auto_fallback`] holds the refusal. It adds no
//! [degradation](ChatExtractionMetadata::degradations): the forced-tool pass
//! reports its own. When the forced-tool attempt fails too before sending
//! (it is refused, or the request carries its own tools), the native
//! refusal is the primary cause: the error is an [`Error::Unsupported`]
//! naming [`RequestField::ResponseFormat`], whose message is about the
//! caller's type and gives the forced-tool failure second ("`Order` cannot
//! be extracted from `m` with its output constrained: native output was
//! refused (...), and forced-tool mode was refused too (...)"). A provider
//! error from the forced-tool call is returned as itself. Either way
//! `auto_fallback` is set in the extraction record. An explicit
//! [`ChatExtractionMode::Native`] never falls back.
//!
//! **Object roots.** Some providers take only an object at the root of a
//! response schema and of tool parameters (OpenAI, on Chat Completions and
//! Responses). Extracting a type whose schema is not an object there, such
//! as an enum or a `Vec`, fails with [`Error::Unsupported`] before sending,
//! in native and forced-tool mode alike. Wrap the value in a struct
//! (`struct Labeled { label: Label }`), or ask a decision model, which
//! answers choice questions natively
//! ([`DecisionProviderExt::decide_as`](crate::DecisionProviderExt::decide_as)).
//!
//! [`ChatExtractExt::extract`] is available on every provider and makes exactly
//! one provider call. [`ChatExtractExt::extract_stream`] streams the same
//! request and parses the value when the stream ends.
//!
//! # Recovery is opt-in
//!
//! An [`ChatExtractor`] runs the same extraction with explicit recovery:
//!
//! - [`ChatExtractor::repair`]: local JSON repair (markdown fences, prose around
//!   the value). No extra provider calls.
//! - [`ChatExtractor::retries`]: when the output cannot be parsed, send the
//!   failure back to the model and ask again. **Each retry is another
//!   billed provider call**, reported in [`ChatExtractionMetadata::passes`].
//!
//! Nothing retries or repairs unless configured to.
//! [`ChatExtractor::parse`] applies the same parsing and repair to a response
//! you already have, without calling a provider.
//!
//! # Bookkeeping
//!
//! Every extraction reports how it ran in an [`ChatExtractionMetadata`]: the
//! mode, the number of provider calls ([`passes`](ChatExtractionMetadata::passes)),
//! their token usage summed ([`usage`](ChatExtractionMetadata::usage)), and
//! how far the provider reported enforcing the schema
//! ([`enforcement`](ChatExtractionMetadata::enforcement)). A successful
//! extraction returns it in [`ChatExtracted`]; a failed one returns
//! [`Error::Extract`] with an [`ChatExtractFailure`] that carries the same record
//! plus the last provider response, so the cost of failed passes and the
//! output that could not be used stay inspectable.
//!
//! A provider error during an extraction (a rate limit on a retry, say) is
//! not an extraction failure: it is returned as itself, so retry and
//! fallback policies still see `RateLimited` or `Timeout`. The record of the
//! passes made before it travels in the error's context; read it with
//! [`ChatExtractionMetadata::from_error`]. When a stream fails mid-way, the
//! record also holds the usage and degradations the stream reported before
//! failing; the stream is not counted as a pass, so `passes` can be `0`
//! while `usage` is not.
//!
//! # Output that is not an answer
//!
//! Before parsing, extraction checks how the response finished, and never
//! retries these failures:
//!
//! - A refusal fails with [`ChatExtractError::Refused`]: the model wrote one
//!   instead of the output, or the response finished with
//!   [`FinishReason::Refusal`] or [`FinishReason::ContentFilter`]. Its
//!   `text` is the model's refusal or the provider's explanation, when
//!   there is one, and [`RefusalSource`] says which. Asking again would
//!   pay for the same refusal.
//! - Output cut off at the token limit ([`FinishReason::Length`]) fails
//!   with [`ChatExtractError::Truncated`]: the same limit would cut it off
//!   again. On a reasoning model the limit covers the reasoning too, and
//!   the error says how much of it went to reasoning when the response
//!   reported that; see [Reasoning models](#reasoning-models).
//! - Output cut off by a full context window
//!   ([`FinishReason::ContextWindow`]) fails with
//!   [`ChatExtractError::Truncated`] too, with that finish reason: the input
//!   left too little room, so shorten it (raising the output limit does not
//!   help).
//!
//! Other abnormal finishes (an [`Other`](FinishReason::Other) reason, such
//! as Gemini's `malformed_function_call`) are parsed as usual; when the
//! output cannot be parsed, the [`ChatExtractError::Parse`] message names the
//! finish reason. Empty or whitespace-only output in `Native` mode is
//! [`ChatExtractError::MissingStructuredText`], not a parse failure.
//!
//! # Reasoning models
//!
//! On a reasoning model the output token limit
//! ([`ChatRequest::max_output_tokens`](field@ChatRequest::max_output_tokens),
//! or the provider's default) covers the reasoning as well as the output,
//! wherever the provider counts it so. A model that reasons at length can
//! spend most or all of the limit before writing the value, and the
//! extraction fails with [`ChatExtractError::Truncated`]. When the response
//! reported its reasoning tokens, the error says so ("400 of 400 output
//! tokens went to reasoning").
//!
//! Extraction rarely needs much reasoning. Ask for less on extraction
//! requests, which also costs less, or raise the limit:
//!
//! ```
//! use allama::{ChatRequest, ReasoningEffort};
//!
//! let request = ChatRequest::new("gpt-5-mini")
//!     .user("Ada published the first program in 1843.")
//!     .reasoning(ReasoningEffort::Low)
//!     .max_output_tokens(2_000);
//! # let _ = request;
//! ```
//!
//! # Schema enforcement
//!
//! Both modes ask the provider to constrain the output to the schema of
//! `T`: the response format in `Native` mode and the synthetic tool in
//! `ForcedTool` mode are both strict. Adapters rewrite the schema into
//! their grammar engine's dialect and report, as warnings on the field
//! that carries it, what they could not keep. Extraction reads those
//! warnings into [`ChatExtractionMetadata::enforcement`], a
//! [`SchemaEnforcement`]:
//!
//! - [`Exact`](SchemaEnforcement::Exact): constrained decoding ran over the
//!   schema of `T`, rewritten only in ways that change nothing `T` can
//!   deserialize (annotations removed, a `$ref` and its siblings merged, a
//!   provably disjoint `oneOf` written as `anyOf`, a scalar `const` written
//!   as a one-value `enum`, optional properties made
//!   nullable under strict modes that require every property; the `null`s
//!   that lets the model send are removed before deserializing, see
//!   below);
//! - [`Approximate`](SchemaEnforcement::Approximate): constrained decoding
//!   ran, over a schema the adapter had to widen, or sent to an engine that
//!   does not report what it enforces. An
//!   [`Approximated`](crate::WarningKind::Approximated) warning says what;
//! - [`NotEnforced`](SchemaEnforcement::NotEnforced): strict mode was not
//!   honoured, and an [`Unsupported`](crate::WarningKind::Unsupported)
//!   warning says why (the endpoint has none, the model lacks it, or the
//!   schema uses a construct its grammar cannot express).
//!
//! Only the field the mode relies on counts: the response format in
//! `Native` mode ([`RequestField::ResponseFormat`]) and the tools in
//! `ForcedTool` mode ([`RequestField::Tools`]). A warning about the
//! caller's own tools in `Native` mode leaves the enforcement alone. The
//! counted warnings are in
//! [`degradations`](ChatExtractionMetadata::degradations), reworded in terms of
//! the caller's type: "the schema of `Order`" for the synthetic tool or
//! the response schema, and `Order.attributes` for a JSON pointer such as
//! `/properties/attributes` (the response keeps the adapter's own words).
//!
//! An enforcement below `Exact` is also a warning event (with the
//! `warning-events` feature): an `Approximated` or `Unsupported` warning on
//! the counted field, coded `chat.extraction.<enforcement>.<Type>`
//! (`chat.extraction.not_enforced.Order`), so it is logged at `WARN` once per
//! provider, model and type, and at `DEBUG` after that.
//!
//! When strict mode cannot be honoured, the request's
//! [`UnsupportedPolicy`] decides. Adapters apply it through
//! [`apply_strict_policy`](crate::adapter::schema::apply_strict_policy), the one
//! place that decides between `NotEnforced` and failure for a schema the
//! grammar cannot express:
//!
//! - [`UnsupportedPolicy::Warn`] (the default): the schema goes out
//!   non-strict with an `Unsupported` warning, and extraction still runs,
//!   reporting `NotEnforced`, so a batch job can check one field instead of
//!   reading response warnings.
//! - [`UnsupportedPolicy::Error`]: the provider call fails with
//!   [`Error::Unsupported`] before anything is sent. It means "constrained
//!   decoding is in force, possibly over an approximated schema, or fail":
//!   an `Approximated` rewrite does not fail. A pipeline that requires
//!   `Exact` sets a [floor](self#enforcement-floor-and-plans). In `Auto` mode a refused native response format first
//!   [falls back](self#automatic-fallback-to-forced-tool-mode) to
//!   forced-tool mode, which fails the same way when it cannot constrain
//!   the output either.
//!
//! ```
//! # #[cfg(feature = "mock")]
//! # async fn example() -> allama::Result<()> {
//! use allama::chat::extract::{ChatExtractExt, ChatExtractionMode, SchemaEnforcement};
//! use allama::mock::MockProvider;
//! use allama::schemars::{self, JsonSchema};
//! use allama::{ChatRequest, UnsupportedPolicy};
//!
//! #[derive(serde::Deserialize, JsonSchema)]
//! struct Person {
//!     name: String,
//! }
//!
//! let provider = MockProvider::with_text(r#"{"name": "Ada"}"#);
//! let request = ChatRequest::new("m")
//!     .user("Ada wrote the first program.")
//!     .with_option(ChatExtractionMode::Native)
//!     .unsupported_policy(UnsupportedPolicy::Error); // constrained or fail
//! let extracted = provider.extract::<Person>(&request).await?;
//! assert_eq!(extracted.metadata.enforcement, SchemaEnforcement::Exact);
//! # Ok(()) }
//! ```
//!
//! The enforcement is the adapter's own report, not a verified grammar
//! guarantee: an adapter that ignores strict mode without saying so (or a
//! mock) reports `Exact`. Even `Exact` does not guarantee every value: a
//! refusal or output cut off at the token limit is not schema output, and
//! some engines do not guarantee the capitalisation of string `enum` or
//! `const` values (Anthropic, for one). The output is deserialized into `T`
//! either way, so a value that does not match still fails with
//! [`ChatExtractError::Parse`].
//!
//! # Enforcement floor and plans
//!
//! [`ChatExtractor::plan`] tells what an extraction would do before it sends
//! anything: the mode (after any automatic fallback), the enforcement its
//! first pass would report, and how much of the call the provider's
//! validation covered ([`ChatExtractionPlan`]). It runs the provider's
//! validation on a schema-only copy of the request.
//!
//! [`ChatExtractor::min_enforcement`] sets a floor, off by default. With
//! [`ValidationCoverage::Encoding`] (every first-party adapter) a plan
//! below it fails before sending with [`ChatExtractError::EnforcementBelow`];
//! otherwise each pass is checked, and the failure carries the response
//! and its usage, so it is a paid call.
//! [`Approximate`](SchemaEnforcement::Approximate) is the practical
//! portable floor: Anthropic removes `minimum` and `maximum`, so a type
//! with an unsigned integer field is never `Exact` there.
//!
//! What adapters report for typical types (the extraction-plan tests of
//! each adapter):
//!
//! | Adapter | Mode | Strings | Unsigned integer | Map |
//! |---|---|---|---|---|
//! | OpenAI (both APIs), Azure, Groq, xAI | both | exact | exact | not enforced |
//! | Anthropic | native | exact | approximate | refused (`Auto` falls back) |
//! | Anthropic | forced tool | exact | approximate | not enforced |
//! | Gemini | native | exact | exact | exact |
//! | Gemini | forced tool | not enforced | not enforced | not enforced |
//! | Custom OpenAI-compatible endpoint, OpenRouter, Mistral | both | approximate | approximate | not enforced |
//! | Ollama | native | approximate | approximate | approximate |
//! | Ollama | forced tool | refused | refused | refused |
//! | vLLM | both | approximate | approximate | approximate |
//! | Together, Fireworks, DeepInfra | both | not enforced | not enforced | not enforced |
//! | Workers AI | native | not enforced | not enforced | not enforced |
//!
//! # Undeclared keys
//!
//! serde ignores keys a struct does not declare and fills in defaults, so
//! without constrained decoding a model that misnames a key (`e_mail` for
//! `email`) or nests one at the wrong level yields a value that
//! deserializes and is wrong: the `Option` or `#[serde(default)]` field it
//! meant is empty. On every pass whose enforcement is not `Exact` (and in
//! [`ChatExtractor::parse`], which cannot know), extraction therefore walks the
//! output against the schema of `T` before deserializing, and rejects a
//! key that no schema there declares, with an [`ChatExtractError::Parse`] that
//! names the key and its path (``unknown field `e_mail`, expected one of
//! `email`, `phone` `` at `contact.e_mail`). The rejection is retryable:
//! [`ChatExtractor::retries`] sends it back to the model, and
//! [`ChatExtractionMetadata::undeclared_key_rejections`] counts the passes it
//! cost. The raw output is kept.
//!
//! - **Stricter than JSON Schema and than other libraries.** An object
//!   without `additionalProperties` is treated as closed. JSON Schema
//!   leaves it open, and Pydantic (`extra='ignore'`), Zod and serde ignore
//!   unknown keys by default. With `#[serde(deny_unknown_fields)]`, schemars
//!   writes `additionalProperties: false` and serde rejects the key too, so
//!   the two agree.
//! - **Exempt:** maps (`HashMap`, `BTreeMap`) and structs with a
//!   `#[serde(flatten)]` map (`additionalProperties`), `patternProperties`,
//!   and open schemas (`serde_json::Value`, `{}`, `true`, an object
//!   schema without `properties`).
//! - **Unions:** a key counts as undeclared only when no `anyOf`/`oneOf`
//!   branch that may apply declares it; when the walk cannot pick one
//!   branch, it checks the key against all of them. A key some branch
//!   declares passes even when serde then picks another branch of an
//!   `untagged` enum.
//! - **Aliases:** `#[serde(alias)]` names are not in the schema, so a key
//!   written under an alias is rejected (and retried).
//! - **Opt out** with [`ChatExtractor::allow_undeclared_keys`].
//!
//! The check does not close the remaining gap: a schema sent without
//! strict mode lets the model leave out `Option` and `#[serde(default)]`
//! fields, so a `NotEnforced` pass can still turn "the model did not
//! decide" into a default value. Callers who care set a
//! [floor](self#enforcement-floor-and-plans).
//!
//! # Target types
//!
//! `T` implements `serde::Deserialize` and `schemars::JsonSchema`. Derive the
//! latter through the [`allama::schemars`](crate::schemars) re-export
//! (`use allama::schemars::{self, JsonSchema};`) so it matches the
//! `schemars` version `allama` was built with.
//!
//! Strict modes that require every property make optional properties
//! nullable, so the model may send `null` for a field it would have left
//! out. Before deserializing, extraction removes those stand-in `null`s
//! ([`normalize_strict_nulls`](crate::adapter::schema::normalize_strict_nulls)): a
//! `null` for an optional property whose schema in `T` admits no `null`.
//! `#[serde(default)]` fields then get their default, while `Option<T>` and
//! `serde_json::Value` keep their `null`. A `null` for a required property
//! or a map entry is kept, so serde reports it ("invalid type: null").
//! [`ChatExtractor::parse`] normalizes too.
//!
//! # Validating values
//!
//! A schema says what the model may send, and not every engine enforces
//! all of it (Anthropic removes `minimum` and `maximum`, see the
//! [table](self#enforcement-floor-and-plans)). To be sure of a rule,
//! check it while deserializing: a newtype with `#[serde(try_from = ..)]`
//! rejects a value that breaks it, and its error becomes the
//! [`ChatExtractError::Parse`] message. With [`ChatExtractor::retries`], that
//! message is the feedback the model gets, so say what is wrong in it.
//! With `try_from`, schemars writes the schema of the type converted from
//! (`u8` here), and a `range` attribute on the newtype's field is not
//! applied; add the rule with `#[schemars(extend(..))]` on the newtype, so
//! the schema the model reads states it too:
//!
//! ```
//! # #[cfg(feature = "mock")]
//! # tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async {
//! use allama::chat::extract::ChatExtractor;
//! use allama::mock::MockProvider;
//! use allama::schemars::{self, JsonSchema};
//! use allama::chat::extract::ChatExtractionMode;
//! use allama::ChatRequest;
//!
//! /// A rating from 1 to 5.
//! #[derive(Debug, serde::Deserialize, JsonSchema)]
//! #[serde(try_from = "u8")]
//! #[schemars(extend("minimum" = 1, "maximum" = 5))]
//! struct Rating(u8);
//!
//! impl TryFrom<u8> for Rating {
//!     type Error = String;
//!
//!     fn try_from(value: u8) -> Result<Self, Self::Error> {
//!         if (1..=5).contains(&value) {
//!             Ok(Self(value))
//!         } else {
//!             Err(format!("a rating is 1 to 5, not {value}"))
//!         }
//!     }
//! }
//!
//! #[derive(serde::Deserialize, JsonSchema)]
//! struct Review {
//!     rating: Rating,
//! }
//!
//! let provider = MockProvider::build(|b| b.text(r#"{"rating": 7}"#).text(r#"{"rating": 4}"#));
//! let request = ChatRequest::new("m")
//!     .user("Rate this review: 'Great, but slow.'")
//!     .with_option(ChatExtractionMode::Native);
//! let extracted = ChatExtractor::new()
//!     .retries(1)
//!     .extract::<_, Review>(&provider, &request)
//!     .await?;
//! assert_eq!(extracted.value.rating.0, 4);
//! // The schema carries the rule.
//! assert_eq!(schemars::schema_for!(Rating).get("maximum"), Some(&5.into()));
//! // The model was told what was wrong.
//! let feedback = provider.last_request().unwrap().messages.last().unwrap().text();
//! assert!(feedback.unwrap().contains("a rating is 1 to 5, not 7"));
//! # Ok::<(), allama::Error>(()) }).unwrap();
//! ```

use std::fmt;
use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures_core::Stream;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::chat::stream::{StreamBlockType, StreamCollector};
use crate::error::{EXTRACTION_METADATA_KEY, Withheld};
use crate::macros::enum_names;
use crate::provider_data::{ALLAMA_NAMESPACE, DECLINE_EXPLANATION_KEY};
use crate::{
    CapabilitySupport, ChatCapability, ChatProvider, ChatRequest, ChatResponse, Error,
    FinishReason, Message, ProviderOptions, RequestField, ResponseFormat, Result, StreamEvent,
    Subject, ToolChoice, ToolDefinition, Usage, Warning, WarningKind,
};
#[cfg(doc)]
use crate::{UnsupportedPolicy, ValidationCoverage};

mod enforcement;

pub use enforcement::ChatExtractionPlan;
use enforcement::Target;

const EXTRACTION_TOOL: &str = "submit_structured_output";

/// Structured extraction for every chat provider.
pub trait ChatExtractExt: ChatProvider {
    /// Extract a `T` from the response to `request`, in one provider call.
    ///
    /// The extraction mode is read from the request options (see
    /// [`ChatExtractionMode`]) and removed from the derived request. For repair
    /// or retries, use an [`ChatExtractor`].
    ///
    /// # Errors
    ///
    /// Returns provider errors as-is, with the extraction's record attached
    /// ([`ChatExtractionMetadata::from_error`]), and extraction failures as
    /// [`Error::Extract`], whose [`ChatExtractFailure`] carries the response
    /// that could not be used.
    fn extract<'a, T>(
        &'a self,
        request: &'a ChatRequest,
    ) -> impl Future<Output = Result<ChatExtracted<T>>> + Send + 'a
    where
        Self: Sync,
        T: JsonSchema + DeserializeOwned + Send + 'a,
    {
        async move { ChatExtractor::new().extract(self, request).await }
    }

    /// Stream the extraction request for `request`.
    ///
    /// The returned stream yields the provider's events unchanged and keeps
    /// the raw structured output seen so far
    /// ([`ChatExtractionStream::partial_output`]); call
    /// [`ChatExtractionStream::finish`] after it ends to parse the value. Makes
    /// one provider call.
    ///
    /// # Errors
    ///
    /// Fails like [`extract`](Self::extract) before the stream opens.
    fn extract_stream<'a, T>(
        &'a self,
        request: &'a ChatRequest,
    ) -> impl Future<Output = Result<ChatExtractionStream<Self::Stream, T>>> + Send + 'a
    where
        Self: Sync,
        T: JsonSchema + Send + 'a,
    {
        async move { ChatExtractor::new().extract_stream(self, request).await }
    }
}

impl<P: ChatProvider + ?Sized> ChatExtractExt for P {}

/// How structured output is obtained.
///
/// Attach it to a request with `request.with_option(ChatExtractionMode::Native)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ChatExtractionMode {
    /// Choose from the provider's reported capabilities (see the
    /// [module docs](self)).
    #[default]
    Auto,
    /// Use the provider's JSON Schema response format.
    Native,
    /// Force a call to a synthetic tool and read its arguments.
    ForcedTool,
}

enum_names!(ChatExtractionMode, "extraction mode", {
    Auto => "auto",
    Native => "native",
    ForcedTool => "forced_tool",
});

impl ProviderOptions for ChatExtractionMode {
    const NAMESPACE: &'static str = "allama.chat.extraction";
}

/// A typed extraction result.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ChatExtracted<T> {
    /// The parsed value.
    pub value: T,
    /// The provider response it was parsed from (the last one, after
    /// retries).
    pub response: ChatResponse,
    /// How the extraction ran, including the usage of every pass.
    pub metadata: ChatExtractionMetadata,
}

/// How an extraction ran: its mode and the record of its passes.
///
/// Reported on success ([`ChatExtracted::metadata`]) and on failure
/// ([`ChatExtractFailure::metadata`]).
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ChatExtractionMetadata {
    /// The mode that was used.
    pub mode: ChatExtractionMode,
    /// Number of provider calls made by the extraction (1 plus retries).
    ///
    /// `0` when it failed before calling the provider, and for
    /// [`ChatExtractor::parse`], which parses a response the caller already
    /// has. A call that failed with a provider error returned no response
    /// and is not counted (see [`from_error`](Self::from_error)), even a
    /// stream that failed mid-way, whose usage so far is still in
    /// [`usage`](Self::usage).
    pub passes: u32,
    /// The value was parsed after local repair.
    pub repaired: bool,
    /// How far the provider reported constraining the output to the schema
    /// of `T`, from the warnings it reported on the field the mode relies
    /// on (see [`SchemaEnforcement`] and the
    /// [module docs](self#schema-enforcement)). The least enforced pass
    /// wins.
    ///
    /// This reflects the adapter's own report, not a verified grammar
    /// guarantee. [`NotEnforced`](SchemaEnforcement::NotEnforced) when no
    /// pass was made, including for [`ChatExtractor::parse`], which cannot know
    /// what the request asked for.
    #[serde(default)]
    pub enforcement: SchemaEnforcement,
    /// The warnings counted for [`enforcement`](Self::enforcement), from
    /// every pass and from a stream that failed mid-way, each distinct
    /// warning once: the warnings about the
    /// response format ([`RequestField::ResponseFormat`]) in `Native` mode,
    /// or about the tools ([`RequestField::Tools`]) in `ForcedTool` mode.
    ///
    /// The response keeps every warning ([`ChatResponse::warnings`]),
    /// including those about other fields, which are not counted.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub degradations: Vec<Warning>,
    /// Token usage summed over every pass, including the passes whose
    /// output could not be used (see [`Usage`]'s `Add`). `None` when nothing
    /// reported usage.
    ///
    /// It also holds the usage a stream reported before it failed mid-way
    /// ([`ChatExtractionStream::finish`]). That stream returned no response and
    /// is not a pass, so the record of a failed stream can pair
    /// [`passes`](Self::passes) `0` with billed usage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// The refusal that made an [`Auto`](ChatExtractionMode::Auto) extraction
    /// switch from native output to forced-tool mode, when it did (see the
    /// [module docs](self#automatic-fallback-to-forced-tool-mode)).
    /// [`mode`](Self::mode) is then [`ForcedTool`](ChatExtractionMode::ForcedTool).
    ///
    /// The refused attempt sent nothing and is not counted in
    /// [`passes`](Self::passes).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auto_fallback: Option<AutoFallback>,
    /// How many passes were rejected for a key the schema of `T` does not
    /// declare (see [undeclared keys](self#undeclared-keys)), whether the
    /// extraction asked again or failed. Each one asked again is a billed
    /// pass, so this is the cost of the check.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub undeclared_key_rejections: u32,
}

impl ChatExtractionMetadata {
    /// The record of an extraction in `mode` that has made no provider calls
    /// yet.
    #[must_use]
    pub fn new(mode: ChatExtractionMode) -> Self {
        Self {
            mode,
            passes: 0,
            repaired: false,
            enforcement: SchemaEnforcement::NotEnforced,
            degradations: Vec::new(),
            usage: None,
            auto_fallback: None,
            undeclared_key_rejections: 0,
        }
    }

    /// The pass record attached to a provider error that ended an
    /// extraction.
    ///
    /// Extraction returns provider errors unchanged (so
    /// [`Error::is_retryable`], [`Error::retry_after`] and the variant stay
    /// what the provider reported) and attaches the record of the passes
    /// made before the error to its context, under `allama.chat.extraction` in
    /// [`ErrorContext::metadata`](crate::ErrorContext::metadata). It is
    /// attached on every provider error from [`ChatExtractExt::extract`],
    /// [`ChatExtractExt::extract_stream`], [`ChatExtractionStream::finish`] and the
    /// [`ChatExtractor`] methods, including one on the first call
    /// ([`passes`](Self::passes) `0`). For an error that ended a failover it
    /// is on that error's own context, not on the errors it
    /// [fell back from](Error::fallback_from).
    ///
    /// `None` when the error carries no record: it did not come from an
    /// extraction, or its variant has no context ([`Error::Serialization`]).
    /// An [`Error::Extract`] carries its record in
    /// [`ChatExtractFailure::metadata`] instead.
    ///
    /// ```
    /// # #[cfg(feature = "mock")]
    /// # async fn example() {
    /// use allama::chat::extract::{ChatExtractionMetadata, ChatExtractionMode, ChatExtractor};
    /// use allama::mock::{ChatResponseBuilder, MockProvider};
    /// use allama::schemars::{self, JsonSchema};
    /// use allama::{ChatRequest, Error};
    ///
    /// #[derive(Debug, serde::Deserialize, JsonSchema)]
    /// struct Person {
    ///     name: String,
    /// }
    ///
    /// let provider = MockProvider::build(|b| {
    ///     b.response(ChatResponseBuilder::new().text("not json").usage(500, 50).build())
    ///         .error(Error::rate_limited("slow down"))
    /// });
    /// let request = ChatRequest::new("m")
    ///     .user("Ada wrote the first program.")
    ///     .with_option(ChatExtractionMode::Native);
    /// let error = ChatExtractor::new()
    ///     .retries(1)
    ///     .extract::<_, Person>(&provider, &request)
    ///     .await
    ///     .unwrap_err();
    /// assert!(matches!(error, Error::RateLimited { .. })); // not reclassified
    /// let metadata = ChatExtractionMetadata::from_error(&error).unwrap();
    /// assert_eq!(metadata.passes, 1);
    /// assert_eq!(metadata.usage.and_then(|u| u.input_tokens), Some(500));
    /// # }
    /// ```
    #[must_use]
    pub fn from_error(error: &Error) -> Option<Self> {
        let record = error.context()?.metadata.get(EXTRACTION_METADATA_KEY)?;
        Self::deserialize(record).ok()
    }

    /// Attach this record to a provider error that ended the extraction,
    /// for [`from_error`](Self::from_error).
    fn attach_to(&self, mut error: Error) -> Error {
        if let Some(context) = error.context_mut()
            && let Ok(record) = serde_json::to_value(self)
        {
            context
                .metadata
                .insert(EXTRACTION_METADATA_KEY.to_owned(), record);
        }
        error
    }

    /// The fallback to forced-tool mode that `error` calls for, when this
    /// extraction may still take it: the caller asked for
    /// [`Auto`](ChatExtractionMode::Auto), it picked
    /// [`Native`](ChatExtractionMode::Native), no pass was made yet, and the
    /// response format was refused: by the adapter before sending, or, when
    /// `Native` was only a guess from an `Unknown` answer (`guessed`), by
    /// the provider itself. A provider's 4xx refusal produced nothing to
    /// bill, so the fallback still never pays twice for the output.
    fn auto_fallback_for(
        &self,
        requested: ChatExtractionMode,
        guessed: bool,
        error: &Error,
    ) -> Option<AutoFallback> {
        use crate::UnsupportedStage::{BeforeSending, Provider};
        let stage = error.unsupported_stage()?;
        let refused_here = stage == BeforeSending || (guessed && stage == Provider);
        if !self.may_fall_back(requested)
            || error.unsupported_field() != Some(RequestField::ResponseFormat)
            || !refused_here
        {
            return None;
        }
        // The record reaches the redacted log view, so it keeps only text an
        // adapter authored (every `Degradations` refusal is).
        let message = if error.message_origin() == crate::MessageOrigin::Authored {
            error.message()
        } else {
            String::new()
        };
        Some(AutoFallback::new(RequestField::ResponseFormat, message).with_stage(stage))
    }

    /// Whether a refusal of the native response format would still switch
    /// this extraction to forced-tool mode.
    fn may_fall_back(&self, requested: ChatExtractionMode) -> bool {
        requested == ChatExtractionMode::Auto
            && self.mode == ChatExtractionMode::Native
            && self.passes == 0
            && self.auto_fallback.is_none()
    }

    /// Switch to forced-tool mode after `fallback`.
    fn fall_back(&mut self, fallback: AutoFallback) {
        self.mode = ChatExtractionMode::ForcedTool;
        self.auto_fallback = Some(fallback);
    }

    /// Count one provider call, add its usage, and record the warnings it
    /// reported on the field the mode relies on.
    ///
    /// Every extraction request asks for strict mode, so the enforcement
    /// follows from the counted warnings alone. The counted warnings are
    /// kept in terms of the caller's type, `label`.
    fn record_pass(&mut self, response: &ChatResponse, label: &str) {
        self.passes = self.passes.saturating_add(1);
        self.record_usage_and_degradations(response, label);
        self.enforcement = SchemaEnforcement::from_degradations(&self.degradations);
    }

    /// Add the usage and degradations a stream reported before it failed.
    ///
    /// The stream returned no response, so it is not counted as a pass, and
    /// [`enforcement`](Self::enforcement) stays as it was.
    fn record_failed_stream(&mut self, partial: &ChatResponse, label: &str) {
        self.record_usage_and_degradations(partial, label);
    }

    fn record_usage_and_degradations(&mut self, response: &ChatResponse, label: &str) {
        if let Some(usage) = &response.usage {
            self.usage = Some(self.usage.take().unwrap_or_default() + usage);
        }
        let field = counted_field(self.mode);
        for warning in response
            .warnings
            .iter()
            .filter(|w| field.is_some_and(|field| is_about(w, field)))
        {
            let warning = enforcement::in_caller_terms(warning, label);
            if !self.degradations.contains(&warning) {
                self.degradations.push(warning);
            }
        }
    }
}

/// Shows the mode, passes and enforcement, and the other fields when they
/// are set.
impl fmt::Debug for ChatExtractionMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self {
            mode,
            passes,
            repaired,
            enforcement,
            degradations,
            usage,
            auto_fallback,
            undeclared_key_rejections,
        } = self;
        let mut debug = f.debug_struct("ChatExtractionMetadata");
        debug
            .field("mode", mode)
            .field("passes", passes)
            .field("enforcement", enforcement);
        if *repaired {
            debug.field("repaired", repaired);
        }
        if !degradations.is_empty() {
            debug.field("degradations", degradations);
        }
        if let Some(usage) = usage {
            debug.field("usage", usage);
        }
        if let Some(auto_fallback) = auto_fallback {
            debug.field("auto_fallback", auto_fallback);
        }
        if *undeclared_key_rejections > 0 {
            debug.field("undeclared_key_rejections", undeclared_key_rejections);
        }
        debug.finish()
    }
}

/// How far the provider reported constraining an extraction's output to
/// the schema of `T` ([`ChatExtractionMetadata::enforcement`]).
///
/// It is derived from the warnings the adapter reported on the field the
/// extraction mode relies on; see the
/// [module docs](self#schema-enforcement). The enum is `#[non_exhaustive]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum SchemaEnforcement {
    /// Constrained decoding ran over the schema of `T`, with only rewrites
    /// that change nothing `T` can deserialize. No warning was counted.
    Exact,
    /// Constrained decoding ran over a rewritten (widened) schema, or over
    /// a schema sent unchecked to an engine that does not report what it
    /// enforces. An `Approximated` warning says what.
    Approximate,
    /// Strict mode was not honoured, and an `Unsupported` warning says why;
    /// or no provider call was made.
    #[default]
    NotEnforced,
}

enum_names!(SchemaEnforcement, "schema enforcement", {
    Exact => "exact",
    Approximate => "approximate",
    NotEnforced => "not_enforced",
});

impl SchemaEnforcement {
    /// Whether this enforcement is at least `floor`: `Exact` satisfies
    /// every floor, `Approximate` satisfies `Approximate` and
    /// `NotEnforced`, and `NotEnforced` only itself.
    ///
    /// ```
    /// use allama::chat::extract::SchemaEnforcement;
    ///
    /// assert!(SchemaEnforcement::Exact.satisfies(SchemaEnforcement::Approximate));
    /// assert!(!SchemaEnforcement::NotEnforced.satisfies(SchemaEnforcement::Approximate));
    /// ```
    #[must_use]
    pub fn satisfies(self, floor: Self) -> bool {
        self.rank() >= floor.rank()
    }

    fn rank(self) -> u8 {
        match self {
            Self::NotEnforced => 0,
            Self::Approximate => 1,
            Self::Exact => 2,
        }
    }

    /// The enforcement the counted warnings of an extraction show: any
    /// `Unsupported` warning means strict mode was not honoured; any other
    /// warning means it ran over an approximation.
    fn from_degradations(warnings: &[Warning]) -> Self {
        if warnings.iter().any(|w| w.kind == WarningKind::Unsupported) {
            Self::NotEnforced
        } else if warnings.is_empty() {
            Self::Exact
        } else {
            Self::Approximate
        }
    }
}

/// Why an [`Auto`](ChatExtractionMode::Auto) extraction switched from native
/// output to forced-tool mode ([`ChatExtractionMetadata::auto_fallback`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct AutoFallback {
    /// The request field that was refused ([`Error::unsupported_field`]):
    /// [`RequestField::ResponseFormat`].
    pub field: RequestField,
    /// The refusal's message, as the adapter wrote it. Empty when the
    /// message was not [authored](Error::authored), since the record is
    /// part of the redacted log view.
    pub message: String,
    /// Where it was refused ([`Error::unsupported_stage`]): before sending,
    /// or by the provider when native output was a guess from an `Unknown`
    /// answer. A record without one was refused before sending.
    #[serde(default = "before_sending")]
    pub stage: crate::UnsupportedStage,
}

fn before_sending() -> crate::UnsupportedStage {
    crate::UnsupportedStage::BeforeSending
}

impl AutoFallback {
    /// A fallback after the adapter refused `field` with `message` before
    /// sending.
    #[must_use]
    pub fn new(field: RequestField, message: impl Into<String>) -> Self {
        Self {
            field,
            message: message.into(),
            stage: crate::UnsupportedStage::BeforeSending,
        }
    }

    /// Set where the refusal happened.
    #[must_use]
    pub fn with_stage(mut self, stage: crate::UnsupportedStage) -> Self {
        self.stage = stage;
        self
    }
}

/// A failed extraction: why it failed, the last provider response, and the
/// record of the passes it made.
///
/// Carried by [`Error::Extract`]:
///
/// ```
/// # #[cfg(feature = "mock")]
/// # async fn example() {
/// use allama::chat::extract::{ChatExtractError, ChatExtractionMode, ChatExtractor};
/// use allama::mock::{ChatResponseBuilder, MockProvider};
/// use allama::schemars::{self, JsonSchema};
/// use allama::{ChatRequest, Error};
///
/// #[derive(Debug, serde::Deserialize, JsonSchema)]
/// struct Person {
///     name: String,
/// }
///
/// let provider = MockProvider::build(|b| {
///     b.response(ChatResponseBuilder::new().text("not json").usage(10, 3).build())
///         .response(ChatResponseBuilder::new().text("still not json").usage(14, 4).build())
/// });
/// let request = ChatRequest::new("m")
///     .user("Ada wrote the first program.")
///     .with_option(ChatExtractionMode::Native);
/// let Err(Error::Extract { failure, .. }) = ChatExtractor::new()
///     .retries(1)
///     .extract::<_, Person>(&provider, &request)
///     .await
/// else {
///     panic!("expected an extraction failure");
/// };
/// assert!(matches!(failure.error(), ChatExtractError::Parse { .. }));
/// assert_eq!(failure.response().and_then(|r| r.text()).as_deref(), Some("still not json"));
/// assert_eq!(failure.metadata().passes, 2);
/// assert_eq!(failure.metadata().usage.as_ref().and_then(|u| u.input_tokens), Some(24));
/// # }
/// ```
///
/// [`Error`]'s serde form includes the response; its redacted logging view
/// ([`Error::as_log`]) does not. Its `Debug` withholds the response, and the
/// model's text in the error (see [`ChatExtractError`]).
#[derive(Clone, Serialize, Deserialize)]
pub struct ChatExtractFailure {
    error: ChatExtractError,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    response: Option<ChatResponse>,
    metadata: ChatExtractionMetadata,
}

impl ChatExtractFailure {
    /// A failure without a response (extraction stopped before or without
    /// one).
    #[must_use]
    pub fn new(error: ChatExtractError, metadata: ChatExtractionMetadata) -> Self {
        Self {
            error,
            response: None,
            metadata,
        }
    }

    /// Attach the response whose output could not be used.
    #[must_use]
    pub fn with_response(mut self, response: ChatResponse) -> Self {
        self.response = Some(response);
        self
    }

    /// Why extraction failed.
    #[must_use]
    pub fn error(&self) -> &ChatExtractError {
        &self.error
    }

    /// The last provider response: the one whose output could not be used.
    /// `None` when extraction failed before calling the provider.
    #[must_use]
    pub fn response(&self) -> Option<&ChatResponse> {
        self.response.as_ref()
    }

    /// How the extraction ran up to the failure: mode, provider calls made,
    /// and their summed usage.
    #[must_use]
    pub fn metadata(&self) -> &ChatExtractionMetadata {
        &self.metadata
    }

    /// Take the last provider response.
    #[must_use]
    pub fn into_response(self) -> Option<ChatResponse> {
        self.response
    }
}

impl fmt::Display for ChatExtractFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.error.fmt(f)
    }
}

impl fmt::Debug for ChatExtractFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = f.debug_struct("ChatExtractFailure");
        debug.field("error", &self.error);
        if self.response.is_some() {
            debug.field("response", &format_args!("<withheld>"));
        }
        debug.field("metadata", &self.metadata).finish()
    }
}

impl std::error::Error for ChatExtractFailure {}

impl From<ChatExtractFailure> for Error {
    fn from(failure: ChatExtractFailure) -> Self {
        Error::extract(failure)
    }
}

/// Runs extraction with explicit, opt-in recovery.
///
/// The default extractor behaves exactly like [`ChatExtractExt::extract`]: one
/// provider call, strict parsing.
///
/// ```
/// # #[cfg(feature = "mock")]
/// # async fn example() -> allama::Result<()> {
/// use allama::ChatRequest;
/// use allama::chat::extract::ChatExtractor;
/// use allama::mock::MockProvider;
/// use allama::schemars::{self, JsonSchema};
///
/// #[derive(serde::Deserialize, JsonSchema)]
/// struct Person {
///     name: String,
/// }
///
/// let provider = MockProvider::with_text("```json\n{\"name\": \"Ada\"}\n```");
/// let request = ChatRequest::new("m")
///     .user("Ada wrote the first program.")
///     .with_option(allama::chat::extract::ChatExtractionMode::Native);
/// let extracted = ChatExtractor::new()
///     .repair(true) // strip the markdown fence locally
///     .retries(1) // ask once more if it still does not parse
///     .extract::<_, Person>(&provider, &request)
///     .await?;
/// assert!(extracted.metadata.repaired);
/// # Ok(()) }
/// ```
///
/// # Extracting from a conversation
///
/// An extraction keeps every field of the request it is given (see
/// [Typical use](self#typical-use)). To extract from a conversation without
/// the rest of the request, start a fresh request from its messages, or
/// clear the tools on a copy. Either way, ask for the value in a final user
/// message: a conversation that ends on the assistant's answer would send
/// that answer as a [prefill](crate::adapter::chat::retain_turns), and several
/// providers refuse a prefill with structured output.
///
/// ```
/// use allama::chat::extract::ChatExtractionMode;
/// use allama::{ChatRequest, ToolDefinition};
///
/// let chat = ChatRequest::new("m")
///     .user("Who wrote the first program?")
///     .assistant("Ada Lovelace, in 1843.")
///     .tools([ToolDefinition::new("lookup", serde_json::json!({"type": "object"}))])
///     .temperature(0.7);
///
/// // Only the model and the conversation, with the instruction last:
/// let fresh = ChatRequest::new(&chat.model)
///     .extend_messages(chat.messages.clone())
///     .user("Extract the person from this conversation.")
///     .with_option(ChatExtractionMode::ForcedTool);
///
/// // Everything but the tools:
/// let mut without_tools = chat
///     .clone()
///     .user("Extract the person from this conversation.")
///     .with_option(ChatExtractionMode::ForcedTool);
/// without_tools.tools.clear();
/// without_tools.tool_choice = None;
/// # assert!(fresh.tools.is_empty() && fresh.temperature.is_none());
/// # assert_eq!(without_tools.temperature, Some(0.7));
/// # assert!(matches!(fresh.messages.last(), Some(allama::Message::User(_))));
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct ChatExtractor {
    repair: bool,
    retries: u32,
    allow_undeclared_keys: bool,
    min_enforcement: Option<SchemaEnforcement>,
}

impl ChatExtractor {
    /// An extractor with no recovery.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Repair malformed output locally before giving up: strip markdown code
    /// fences and prose around the JSON value. Makes no provider calls;
    /// [`ChatExtractionMetadata::repaired`] reports when it was needed.
    ///
    /// When the repaired value is valid JSON but does not fit `T`, the
    /// [`ChatExtractError::Parse`] message (and the retry feedback) is its error,
    /// such as an unknown enum variant, not the error of the unrepaired
    /// text.
    #[must_use]
    pub fn repair(mut self, enabled: bool) -> Self {
        self.repair = enabled;
        self
    }

    /// When the output cannot be used (unparseable, no structured text, or
    /// the extraction tool was not called once), send the failure back to
    /// the model and ask again, up to `retries` times.
    ///
    /// The feedback names the failure: output that is not valid JSON ("it
    /// was not valid JSON: ...") is told apart from valid JSON that does not
    /// fit the schema ("it did not match the schema at `items[2].price`:
    /// ..."). The path ([`ChatExtractError::Parse`]'s `path`) names the field
    /// to fix; it is found by deserializing again, only after a failure.
    ///
    /// Each retry is one more provider call, billed and with its own
    /// latency; [`ChatExtractionMetadata`] counts the calls and sums their usage,
    /// on success and on failure. Truncated output
    /// ([`ChatExtractError::Truncated`]) is not retried: the same token limit
    /// would cut it off again. Neither is a refusal
    /// ([`ChatExtractError::Refused`]): asking again would pay for the same
    /// refusal.
    ///
    /// A retry replays the failed reply. Each tool call in it is answered
    /// with an ordinary tool result whose text says why it could not be
    /// used, not a failed one ([`ToolMessage::is_error`](crate::ToolMessage::is_error)),
    /// so the retry turn is sent as is on every adapter, including one
    /// whose endpoint has no tool error flag under
    /// [`UnsupportedPolicy::Error`].
    ///
    /// Provider errors are not retried here; use
    /// [`RetryingProvider`](crate::wrappers::RetryingProvider) for those. A
    /// provider error on a retry is returned as itself, not as an extraction
    /// failure, with the passes and usage so far attached: read them with
    /// [`ChatExtractionMetadata::from_error`].
    #[must_use]
    pub fn retries(mut self, retries: u32) -> Self {
        self.retries = retries;
        self
    }

    /// Accept output with keys the schema of `T` does not declare, as serde
    /// does, instead of rejecting it (the default; see
    /// [undeclared keys](self#undeclared-keys)).
    ///
    /// ```
    /// # #[cfg(feature = "mock")]
    /// # async fn example() -> allama::Result<()> {
    /// use allama::chat::extract::{ChatExtractionMode, ChatExtractor};
    /// use allama::mock::{ChatResponseBuilder, MockProvider};
    /// use allama::schemars::{self, JsonSchema};
    /// use allama::{ChatRequest, RequestField, Warning};
    ///
    /// #[derive(serde::Deserialize, JsonSchema)]
    /// struct Person {
    ///     name: String,
    /// }
    ///
    /// // A provider that reports it approximated the schema, so the
    /// // output is checked for undeclared keys.
    /// let approximated = Warning::new(
    ///     allama::WarningKind::Approximated,
    ///     allama::Subject::Field { field: RequestField::ResponseFormat },
    ///     "the grammar engine may not enforce every keyword; the schema was sent unchecked",
    /// );
    /// let provider = MockProvider::with_response(
    ///     ChatResponseBuilder::new()
    ///         .text(r#"{"name": "Ada", "born": 1815}"#)
    ///         .warning(approximated)
    ///         .build(),
    /// );
    /// let request = ChatRequest::new("m").user("Ada").with_option(ChatExtractionMode::Native);
    /// let extracted = ChatExtractor::new()
    ///     .allow_undeclared_keys(true)
    ///     .extract::<_, Person>(&provider, &request)
    ///     .await?;
    /// assert_eq!(extracted.value.name, "Ada");
    /// # Ok(()) }
    /// ```
    #[must_use]
    pub fn allow_undeclared_keys(mut self, allow: bool) -> Self {
        self.allow_undeclared_keys = allow;
        self
    }

    /// Require at least `floor` schema enforcement
    /// ([`ChatExtractionMetadata::enforcement`]), failing with
    /// [`ChatExtractError::EnforcementBelow`] otherwise. Off by default.
    ///
    /// When the provider's validation is a dry run of the call
    /// ([`ValidationCoverage::Encoding`], every first-party adapter), the
    /// floor is checked before sending, on a [plan](Self::plan) of the
    /// extraction, and a plan below it costs nothing. Otherwise
    /// ([`ValidationCoverage::Capabilities`]) the enforcement is known only
    /// from the response, so the floor is checked after each pass, and the
    /// failure carries that pass's response and usage: **it is a paid
    /// call**.
    ///
    /// [`Approximate`](SchemaEnforcement::Approximate) is the practical
    /// portable floor: [`Exact`](SchemaEnforcement::Exact) rules out
    /// providers whose strict mode drops keywords the schema of `T` uses,
    /// such as Anthropic, which removes `minimum` and `maximum`, so any
    /// unsigned integer field is `Approximate` there.
    #[must_use]
    pub fn min_enforcement(mut self, floor: SchemaEnforcement) -> Self {
        self.min_enforcement = Some(floor);
        self
    }

    /// Extract a `T` from the response to `request`. Every field of
    /// `request` is kept except the ones the selected mode replaces
    /// (response format, or tools and tool choice); see
    /// [extracting from a conversation](Self#extracting-from-a-conversation).
    ///
    /// # Errors
    ///
    /// See [`ChatExtractExt::extract`]. After the last retry, the last
    /// extraction failure is returned, with the last response and the usage
    /// of every pass.
    pub async fn extract<P, T>(
        &self,
        provider: &P,
        request: &ChatRequest,
    ) -> Result<ChatExtracted<T>>
    where
        P: ChatProvider + ?Sized,
        T: JsonSchema + DeserializeOwned + Send,
    {
        let target = Target::new::<T>(provider.provider_name(), &request.model);
        let requested = requested_mode(request);
        self.check_floor_before_sending(provider, request, &target, false)?;
        let (mode, guessed) = select_mode(provider, request);
        let mut metadata = ChatExtractionMetadata::new(mode);
        let mut derived = target.derive(request, &metadata)?;
        let mut retries_left = self.retries;
        loop {
            let response = match provider.chat(&derived).await {
                Ok(response) => response,
                Err(error) => {
                    let Some(fallback) = metadata.auto_fallback_for(requested, guessed, &error)
                    else {
                        return Err(target.pre_send_error(error, &metadata));
                    };
                    metadata.fall_back(fallback);
                    derived = target.derive(request, &metadata)?;
                    continue;
                }
            };
            metadata.record_pass(&response, &target.label);
            target.report_enforcement(&metadata);
            if let Some(failure) = self.below_floor(&metadata) {
                return Err(failure.with_response(response).into());
            }
            let mode = metadata.mode;
            let parsed = parse_value::<T>(
                &response,
                mode,
                &target.provider,
                self.repair,
                Some(&target.schema),
                self.rejects_undeclared(&metadata),
            );
            match parsed {
                Ok((value, repaired)) => {
                    metadata.repaired = repaired;
                    return Ok(ChatExtracted {
                        value,
                        response,
                        metadata,
                    });
                }
                Err(rejection) => {
                    if rejection.undeclared {
                        metadata.undeclared_key_rejections += 1;
                    }
                    if retries_left > 0 && rejection.error.worth_asking_again() {
                        retries_left -= 1;
                        ask_again(&mut derived, response, mode, &rejection);
                    } else {
                        return Err(ChatExtractFailure::new(rejection.error, metadata)
                            .with_response(response)
                            .into());
                    }
                }
            }
        }
    }

    /// Parse a `T` from a response you already have, with this extractor's
    /// repair setting. Makes no provider calls.
    ///
    /// Use it for responses obtained outside an extraction call: a batch
    /// result, a recorded response, or a request you built yourself with a
    /// [`ResponseFormat::JsonSchema`]. `mode` says where the output is: the
    /// response text ([`Native`](ChatExtractionMode::Native)) or the arguments
    /// of the single `submit_structured_output` tool call
    /// ([`ForcedTool`](ChatExtractionMode::ForcedTool)).
    /// [`Auto`](ChatExtractionMode::Auto) has nothing to choose from and fails
    /// with [`ChatExtractError::Unsupported`].
    ///
    /// Strict `null`s are normalized against the schema of `T`, as in
    /// [`extract`](Self::extract); a parse failure keeps the text the model
    /// sent.
    ///
    /// The response is returned in [`ChatExtracted`] or in the failure. The
    /// metadata reports no passes, no usage, and no degradations (they are
    /// on the response), and
    /// [`enforcement`](ChatExtractionMetadata::enforcement) is
    /// [`NotEnforced`](SchemaEnforcement::NotEnforced): whether the request
    /// asked for strict mode is unknown here.
    ///
    /// ```
    /// use allama::ChatResponse;
    /// use allama::chat::extract::{ChatExtractionMode, ChatExtractor};
    /// use allama::schemars::{self, JsonSchema};
    ///
    /// #[derive(serde::Deserialize, JsonSchema)]
    /// struct Person {
    ///     name: String,
    /// }
    ///
    /// let response = ChatResponse::new(vec![allama::ContentBlock::text(
    ///     "```json\n{\"name\": \"Ada\"}\n```",
    /// )]);
    /// let extracted = ChatExtractor::new()
    ///     .repair(true)
    ///     .parse::<Person>(response, ChatExtractionMode::Native)?;
    /// assert_eq!(extracted.value.name, "Ada");
    /// assert!(extracted.metadata.repaired);
    /// # Ok::<(), allama::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Extract`] when the output cannot be used.
    pub fn parse<T: JsonSchema + DeserializeOwned>(
        &self,
        response: ChatResponse,
        mode: ChatExtractionMode,
    ) -> Result<ChatExtracted<T>> {
        let provider = response.provider.as_deref().unwrap_or("unknown").to_owned();
        let metadata = ChatExtractionMetadata::new(mode);
        if mode == ChatExtractionMode::Auto {
            return Err(ChatExtractFailure::new(
                ChatExtractError::Unsupported { mode, provider },
                metadata,
            )
            .with_response(response)
            .into());
        }
        let model = response.model.clone().unwrap_or_default();
        let target = Target::new::<T>(&provider, &model);
        let reject_undeclared = self.rejects_undeclared(&metadata);
        finish_extraction(response, metadata, &target, self.repair, reject_undeclared)
    }

    /// Stream the extraction of a `T` from the response to `request`. Local
    /// repair applies when the stream is finished; retries do not (a retry
    /// would be a new stream, which is the caller's call to make).
    ///
    /// # Errors
    ///
    /// See [`ChatExtractExt::extract_stream`].
    pub async fn extract_stream<P, T>(
        &self,
        provider: &P,
        request: &ChatRequest,
    ) -> Result<ChatExtractionStream<P::Stream, T>>
    where
        P: ChatProvider + ?Sized,
        T: JsonSchema,
    {
        let target = Target::new::<T>(provider.provider_name(), &request.model);
        let requested = requested_mode(request);
        self.check_floor_before_sending(provider, request, &target, true)?;
        let (mode, guessed) = select_mode(provider, request);
        let mut metadata = ChatExtractionMetadata::new(mode);
        loop {
            let derived = target.derive(request, &metadata)?;
            let mut stream = match provider.chat_stream(&derived).await {
                Ok(stream) => Box::pin(stream),
                Err(error) => {
                    let Some(fallback) = metadata.auto_fallback_for(requested, guessed, &error)
                    else {
                        return Err(target.pre_send_error(error, &metadata));
                    };
                    metadata.fall_back(fallback);
                    continue;
                }
            };
            // The stream contract allows a refusal as the first item too, so
            // while a fallback is possible, read that item before handing the
            // stream on.
            let mut first = None;
            if metadata.may_fall_back(requested) {
                let item = std::future::poll_fn(|cx| stream.as_mut().poll_next(cx)).await;
                if let Some(Err(error)) = &item
                    && let Some(fallback) = metadata.auto_fallback_for(requested, guessed, error)
                {
                    metadata.fall_back(fallback);
                    continue;
                }
                first = Some(item);
            }
            return Ok(ChatExtractionStream {
                inner: stream,
                first,
                collector: StreamCollector::new(),
                partial: String::new(),
                tool_index: None,
                mode: metadata.mode,
                auto_fallback: metadata.auto_fallback,
                target,
                extractor: *self,
                done: false,
                _value: PhantomData,
            });
        }
    }
}

/// A streamed extraction: the provider's events, then a parsed value.
///
/// Created by [`ChatExtractExt::extract_stream`]. Read the stream to the end,
/// then call [`finish`](Self::finish). It ends after the provider's first
/// error, as every [`ChatStream`](crate::ChatStream) does.
pub struct ChatExtractionStream<S, T> {
    inner: Pin<Box<S>>,
    /// The first item, when it was read before the stream was handed on
    /// (to check for a refusal); `Some(None)` when the stream had ended.
    first: Option<Option<Result<StreamEvent>>>,
    collector: StreamCollector,
    partial: String,
    tool_index: Option<usize>,
    mode: ChatExtractionMode,
    auto_fallback: Option<AutoFallback>,
    /// The schema of `T` and how messages name it.
    target: Target,
    /// Repair, undeclared keys and the floor.
    extractor: ChatExtractor,
    done: bool,
    _value: PhantomData<fn() -> T>,
}

impl<S, T> ChatExtractionStream<S, T> {
    /// The mode the extraction uses: `ForcedTool` after an
    /// [automatic fallback](self#automatic-fallback-to-forced-tool-mode).
    #[must_use]
    pub fn mode(&self) -> ChatExtractionMode {
        self.mode
    }

    /// The raw structured output received so far: the response text in
    /// `Native` mode, the extraction tool's arguments in `ForcedTool` mode.
    ///
    /// It is usually an incomplete JSON document; parse it with a
    /// partial-JSON parser to show progress.
    #[must_use]
    pub fn partial_output(&self) -> &str {
        &self.partial
    }

    /// Parse the value from the collected response.
    ///
    /// # Errors
    ///
    /// Returns the stream's error when it failed or ended early, with the
    /// extraction's record attached ([`ChatExtractionMetadata::from_error`]).
    /// The failed stream is not counted as a pass, but the usage and the
    /// [degradations](ChatExtractionMetadata::degradations) it reported before
    /// failing are in the record, so `passes` can be `0` while `usage`
    /// holds the tokens already billed. Returns [`Error::Extract`] (with the
    /// collected response) when the output cannot be used.
    pub fn finish(self) -> Result<ChatExtracted<T>>
    where
        T: DeserializeOwned,
    {
        let mut metadata = ChatExtractionMetadata::new(self.mode);
        metadata.auto_fallback = self.auto_fallback;
        // A stream that fails mid-way returns no response, but what it
        // reported before failing (usage, warnings) is still the caller's.
        let mut failed = metadata.clone();
        failed.record_failed_stream(self.collector.response_metadata(), &self.target.label);
        let response = self
            .collector
            .finish()
            .map_err(|error| failed.attach_to(error))?;
        metadata.record_pass(&response, &self.target.label);
        self.target.report_enforcement(&metadata);
        if let Some(failure) = self.extractor.below_floor(&metadata) {
            return Err(failure.with_response(response).into());
        }
        let reject_undeclared = self.extractor.rejects_undeclared(&metadata);
        finish_extraction(
            response,
            metadata,
            &self.target,
            self.extractor.repair,
            reject_undeclared,
        )
    }

    fn observe(&mut self, event: &StreamEvent) {
        match event {
            StreamEvent::TextDelta { text, .. } if self.mode == ChatExtractionMode::Native => {
                self.partial.push_str(text);
            }
            StreamEvent::BlockStart {
                index,
                block_type: StreamBlockType::ToolCall,
                name: Some(name),
                ..
            } if name == EXTRACTION_TOOL && self.tool_index.is_none() => {
                self.tool_index = Some(*index);
            }
            StreamEvent::ToolCallDelta { index, arguments } if self.tool_index == Some(*index) => {
                self.partial.push_str(arguments);
            }
            _ => {}
        }
    }
}

/// Counts and state, never content.
impl<S, T> fmt::Debug for ChatExtractionStream<S, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChatExtractionStream")
            .field("mode", &self.mode)
            .field("collector", &self.collector)
            .field("partial_output_len", &self.partial.len())
            .field("done", &self.done)
            .finish_non_exhaustive()
    }
}

impl<S, T> Stream for ChatExtractionStream<S, T>
where
    S: Stream<Item = Result<StreamEvent>>,
{
    type Item = Result<StreamEvent>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        if this.done {
            return Poll::Ready(None);
        }
        let poll = match this.first.take() {
            Some(item) => Poll::Ready(item),
            None => this.inner.as_mut().poll_next(cx),
        };
        match &poll {
            Poll::Ready(Some(Ok(event))) => {
                this.observe(event);
                if let Err(error) = this.collector.push_ref(event) {
                    this.done = true;
                    this.collector.set_terminal_error(error.clone());
                    return Poll::Ready(Some(Err(error)));
                }
            }
            // A chat stream ends after its first error.
            Poll::Ready(Some(Err(error))) => {
                this.done = true;
                this.collector.set_terminal_error(error.clone());
            }
            Poll::Ready(None) => this.done = true,
            Poll::Pending => {}
        }
        poll
    }
}

/// Why an extraction failed after (or before) calling the provider.
///
/// The enum and its variants are `#[non_exhaustive]`: match variants with
/// `..`.
///
/// `Debug` shows the length of the model's text (a parse failure's raw
/// output, message and path, and a refusal's text), never the text itself;
/// `Display` shows all but the raw output.
#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ChatExtractError {
    /// No mode could be selected: the provider's capability answers rule out
    /// both native output and forced tool calls (or [`ChatExtractor::parse`] was
    /// given [`Auto`](ChatExtractionMode::Auto)).
    #[non_exhaustive]
    Unsupported {
        /// Requested mode.
        mode: ChatExtractionMode,
        /// Provider name.
        provider: String,
    },
    /// The request carries tools of its own, which extraction refuses in
    /// every mode before sending. Clear `tools` and `tool_choice`, or start
    /// a fresh request from the messages; see
    /// [extracting from a conversation](ChatExtractor#extracting-from-a-conversation).
    #[non_exhaustive]
    ToolConflict {
        /// Provider name.
        provider: String,
    },
    /// The response had no text to parse (`Native`).
    #[non_exhaustive]
    MissingStructuredText {
        /// Provider name.
        provider: String,
    },
    /// The response had no extraction tool call (`ForcedTool`).
    #[non_exhaustive]
    MissingToolCall {
        /// Provider name.
        provider: String,
    },
    /// The response had several extraction tool calls.
    #[non_exhaustive]
    UnexpectedToolCallCount {
        /// Provider name.
        provider: String,
        /// Number of calls.
        count: usize,
    },
    /// The model stopped at the output token limit
    /// ([`FinishReason::Length`]), or filled the model's context window
    /// ([`FinishReason::ContextWindow`]), before finishing the structured
    /// output. `finish_reason` says which.
    ///
    /// Checked before parsing and never retried: asking again under the
    /// same limit would cut the output off again. At the output token
    /// limit, raise `max_output_tokens` or shorten the expected output; at
    /// the context window, shorten the input (truncate or summarize the
    /// conversation), since the output cap is not what stopped it.
    ///
    /// On a reasoning model the limit covers reasoning too (see
    /// [`ChatRequest::max_output_tokens`](field@ChatRequest::max_output_tokens)),
    /// so reasoning may have used up part or all of it. When the response
    /// reported its reasoning tokens, `Display` says how many of the output
    /// tokens went to reasoning; lowering the reasoning effort then leaves
    /// more of the limit for the output.
    #[non_exhaustive]
    Truncated {
        /// Mode used.
        mode: ChatExtractionMode,
        /// Provider name.
        provider: String,
        /// What cut the output off: [`FinishReason::Length`] or
        /// [`FinishReason::ContextWindow`].
        #[serde(default = "output_limit")]
        finish_reason: FinishReason,
        /// Output tokens the response reported, reasoning included
        /// ([`Usage::output_tokens`]); `None` when it did not report them.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        output_tokens: Option<u64>,
        /// Reasoning tokens the response reported, a subset of
        /// `output_tokens` ([`Usage::reasoning_tokens`]); `None` when it did
        /// not report them.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reasoning_tokens: Option<u64>,
    },
    /// The model or the provider declined to produce the output: the model
    /// wrote a refusal instead, or the response finished with
    /// [`FinishReason::Refusal`] or [`FinishReason::ContentFilter`].
    ///
    /// Checked before parsing and never retried: asking again would pay for
    /// the same refusal. The last response, with any partial output and the
    /// provider's own decline data, is on the [`ChatExtractFailure`].
    #[non_exhaustive]
    Refused {
        /// Mode used.
        mode: ChatExtractionMode,
        /// Provider name.
        provider: String,
        /// The response's finish reason, when it reported one.
        finish_reason: Option<FinishReason>,
        /// What the model or the provider said, when it said anything: the
        /// model's refusal, or the provider's explanation of its decline
        /// (`source` says which). Upstream text: it is left out of the
        /// redacted [`ErrorLog`](crate::ErrorLog).
        text: Option<String>,
        /// Where `text` came from.
        source: RefusalSource,
    },
    /// The schema enforcement is below the floor the extractor requires
    /// ([`ChatExtractor::min_enforcement`]).
    ///
    /// Before sending when the provider's validation is a dry run of the
    /// call; after a pass otherwise, with its response on the
    /// [`ChatExtractFailure`]. Never retried: asking again gets the same
    /// enforcement.
    #[non_exhaustive]
    EnforcementBelow {
        /// The floor.
        required: SchemaEnforcement,
        /// What the provider reported, or would report.
        achieved: SchemaEnforcement,
        /// Why, from the counted degradations, in terms of the caller's
        /// type.
        reason: String,
    },
    /// The output was not valid JSON or did not match the schema of `T`,
    /// including a key the schema does not declare (see
    /// [undeclared keys](self#undeclared-keys)).
    #[non_exhaustive]
    Parse {
        /// Mode used.
        mode: ChatExtractionMode,
        /// Provider name.
        provider: String,
        /// Raw output: the text the model sent, before repair or `null`
        /// normalization.
        raw: String,
        /// Parse error message.
        ///
        /// With [`ChatExtractor::repair`], this is the repaired value's error
        /// when the repair found valid JSON (it reached `T`, so its error
        /// says what is wrong), and the raw output's error otherwise. When
        /// the response finished abnormally for a reason other than a
        /// refusal or the token limit (an [`Other`](FinishReason::Other)
        /// reason), the message ends with it, as in "(finish reason
        /// `malformed_function_call`)".
        message: String,
        /// Where in the output the value that does not fit `T` is, such as
        /// `items[2].price`. The retry feedback names it, so the model
        /// knows which field to fix.
        ///
        /// `None` for output that is not valid JSON and for a failure at
        /// the root. Only the first error is reported, and the path stops
        /// at an `untagged` enum or a `flatten`ed field (serde buffers
        /// their content, so the path names the enum or the struct holding
        /// the `flatten`).
        path: Option<String>,
    },
}

impl ChatExtractError {
    /// The raw output, for parse failures.
    #[must_use]
    pub fn raw_output(&self) -> Option<&str> {
        match self {
            Self::Parse { raw, .. } => Some(raw),
            _ => None,
        }
    }

    /// The message for the redacted [`ErrorLog`](crate::ErrorLog): the
    /// `Display` text, without a parse failure's detail, which may quote the
    /// model's output.
    pub(crate) fn log_message(&self) -> String {
        match self {
            Self::Parse { mode, provider, .. } => {
                format!("{provider}: failed to parse the output using mode {mode:?}")
            }
            Self::Refused {
                provider,
                finish_reason,
                ..
            } => refused_without_text(provider, finish_reason.as_ref()),
            other => other.to_string(),
        }
    }

    /// Whether sending the failure back to the model may help.
    fn worth_asking_again(&self) -> bool {
        matches!(
            self,
            Self::MissingStructuredText { .. }
                | Self::MissingToolCall { .. }
                | Self::UnexpectedToolCallCount { .. }
                | Self::Parse { .. }
        )
    }
}

impl fmt::Display for ChatExtractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported {
                mode: ChatExtractionMode::Auto,
                provider,
            } => write!(
                f,
                "{provider}: the provider reports that the model supports neither structured output nor forced tool calls, so extraction mode Auto has no mode to pick; set ChatExtractionMode explicitly, or override the capability answers with a capability resolver"
            ),
            Self::Unsupported { mode, provider } => write!(
                f,
                "{provider}: extraction mode {mode:?} is not supported; set ChatExtractionMode explicitly or override the capability answers with a capability resolver"
            ),
            Self::ToolConflict { provider } => write!(
                f,
                "{provider}: extraction cannot be combined with request tools; clear `tools` and `tool_choice`, or start a fresh request from the messages"
            ),
            Self::MissingStructuredText { provider } => {
                write!(f, "{provider}: the model did not return structured text")
            }
            Self::MissingToolCall { provider } => write!(
                f,
                "{provider}: the model did not call the extraction tool '{EXTRACTION_TOOL}'"
            ),
            Self::UnexpectedToolCallCount { provider, count } => write!(
                f,
                "{provider}: the model returned {count} extraction tool calls, expected one"
            ),
            Self::Truncated {
                mode,
                provider,
                finish_reason: FinishReason::ContextWindow,
                ..
            } => write!(
                f,
                "{provider}: the model filled its context window before finishing the structured output (mode {mode:?}): shorten the input, a higher output token limit does not help"
            ),
            Self::Truncated {
                mode,
                provider,
                output_tokens,
                reasoning_tokens,
                ..
            } => {
                write!(
                    f,
                    "{provider}: the model stopped at the output token limit before finishing the structured output (mode {mode:?})"
                )?;
                if let Some(share) =
                    crate::usage::reasoning_share(*output_tokens, *reasoning_tokens)
                {
                    write!(
                        f,
                        "; {share}: raise max_output_tokens or lower the reasoning effort"
                    )?;
                }
                Ok(())
            }
            Self::Refused {
                provider,
                text: Some(text),
                ..
            } => write!(f, "{provider}: the model refused: {text}"),
            Self::Refused {
                provider,
                finish_reason,
                ..
            } => f.write_str(&refused_without_text(provider, finish_reason.as_ref())),
            Self::EnforcementBelow {
                required,
                achieved,
                reason,
            } => write!(
                f,
                "schema enforcement `{achieved}` is below the required `{required}`: {reason}"
            ),
            Self::Parse {
                mode,
                provider,
                message,
                path,
                ..
            } => {
                write!(
                    f,
                    "{provider}: failed to parse the output using mode {mode:?}: "
                )?;
                if let Some(path) = path {
                    write!(f, "at `{path}`: ")?;
                }
                f.write_str(message)
            }
        }
    }
}

impl fmt::Debug for ChatExtractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported { mode, provider } => f
                .debug_struct("Unsupported")
                .field("mode", mode)
                .field("provider", provider)
                .finish(),
            Self::ToolConflict { provider } => f
                .debug_struct("ToolConflict")
                .field("provider", provider)
                .finish(),
            Self::MissingStructuredText { provider } => f
                .debug_struct("MissingStructuredText")
                .field("provider", provider)
                .finish(),
            Self::MissingToolCall { provider } => f
                .debug_struct("MissingToolCall")
                .field("provider", provider)
                .finish(),
            Self::UnexpectedToolCallCount { provider, count } => f
                .debug_struct("UnexpectedToolCallCount")
                .field("provider", provider)
                .field("count", count)
                .finish(),
            Self::Truncated {
                mode,
                provider,
                finish_reason,
                output_tokens,
                reasoning_tokens,
            } => {
                let mut debug = f.debug_struct("Truncated");
                debug
                    .field("mode", mode)
                    .field("provider", provider)
                    .field("finish_reason", finish_reason);
                if let Some(output_tokens) = output_tokens {
                    debug.field("output_tokens", output_tokens);
                }
                if let Some(reasoning_tokens) = reasoning_tokens {
                    debug.field("reasoning_tokens", reasoning_tokens);
                }
                debug.finish()
            }
            Self::Refused {
                mode,
                provider,
                finish_reason,
                text,
                source,
            } => {
                let mut debug = f.debug_struct("Refused");
                debug.field("mode", mode).field("provider", provider);
                if let Some(finish_reason) = finish_reason {
                    debug.field("finish_reason", finish_reason);
                }
                if let Some(text) = text {
                    debug.field("text", &Withheld::shown_by_display(text));
                }
                debug.field("source", source).finish()
            }
            Self::EnforcementBelow {
                required,
                achieved,
                reason,
            } => f
                .debug_struct("EnforcementBelow")
                .field("required", required)
                .field("achieved", achieved)
                .field("reason", reason)
                .finish(),
            // The output, the parse error and the path all come from the
            // model's text.
            Self::Parse {
                mode,
                provider,
                raw,
                message,
                path,
            } => {
                let mut debug = f.debug_struct("Parse");
                debug
                    .field("mode", mode)
                    .field("provider", provider)
                    .field("raw", &Withheld::new(raw))
                    .field("message", &Withheld::shown_by_display(message));
                if let Some(path) = path {
                    debug.field("path", &Withheld::shown_by_display(path));
                }
                debug.finish()
            }
        }
    }
}

impl std::error::Error for ChatExtractError {}

/// Where the text of an [`ChatExtractError::Refused`] came from.
///
/// The enum is `#[non_exhaustive]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum RefusalSource {
    /// The model wrote the refusal, in place of the output.
    Model,
    /// The provider declined on the model's behalf and explained why, in
    /// its own decline data (Anthropic's `stop_details`, Gemini's
    /// `finishMessage`; see
    /// [Decline explanations](crate::provider_data#decline-explanations)).
    Provider,
    /// Nothing explains the refusal: only the finish reason reports it.
    None,
}

/// The `Display` of a refusal with no text, also its redacted log line.
fn refused_without_text(provider: &str, finish_reason: Option<&FinishReason>) -> String {
    match finish_reason {
        Some(reason) => format!(
            "{provider}: the model refused to produce the structured output (finish reason `{reason}`)"
        ),
        None => format!("{provider}: the model refused to produce the structured output"),
    }
}

/// The mode the caller asked for, before [`select_mode`] resolves `Auto`.
fn requested_mode(request: &ChatRequest) -> ChatExtractionMode {
    request
        .option::<ChatExtractionMode>()
        .copied()
        .unwrap_or_default()
}

/// Resolve [`Auto`](ChatExtractionMode::Auto) from the provider's answers, in
/// this order:
///
/// 1. `Native` when structured output is `Supported`;
/// 2. `ForcedTool` when tool calls and forced tool choice are both
///    `Supported`;
/// 3. `Native` when structured output is `Unknown` (a guess: the provider
///    decides when the request is sent);
/// 4. `ForcedTool` when neither tool calls nor forced tool choice is
///    `Unsupported` (a guess too);
/// 5. otherwise `Auto`, which fails: the answers rule out both modes.
///
/// A verified mode is preferred over a guessed one, and a guess over a
/// failure: `Unknown` never blocks a request. The flag says whether the mode
/// is step 3's guess, so a provider's own refusal of the response format
/// may still fall back.
fn select_mode<P: ChatProvider + ?Sized>(
    provider: &P,
    request: &ChatRequest,
) -> (ChatExtractionMode, bool) {
    match requested_mode(request) {
        ChatExtractionMode::Auto => {
            use CapabilitySupport::{Supported, Unknown, Unsupported};
            let answer = |c| provider.chat_capability(&request.model, c);
            let native = answer(ChatCapability::StructuredOutput);
            let tools = answer(ChatCapability::ToolCalls);
            let forced = answer(ChatCapability::ForcedToolChoice);
            match (native, tools, forced) {
                (Supported, _, _) => (ChatExtractionMode::Native, false),
                (_, Supported, Supported) => (ChatExtractionMode::ForcedTool, false),
                (Unknown, _, _) => (ChatExtractionMode::Native, true),
                (_, Supported | Unknown, Supported | Unknown) => {
                    (ChatExtractionMode::ForcedTool, false)
                }
                (Unsupported, _, _) => (ChatExtractionMode::Auto, false),
            }
        }
        mode => (mode, false),
    }
}

fn derive_request(
    request: &ChatRequest,
    mode: ChatExtractionMode,
    provider: &str,
    name: &str,
    schema: serde_json::Value,
) -> std::result::Result<ChatRequest, ChatExtractError> {
    if mode != ChatExtractionMode::Auto && !request.tools.is_empty() {
        return Err(ChatExtractError::ToolConflict {
            provider: provider.to_owned(),
        });
    }
    let mut derived = request.clone();
    derived.options.remove::<ChatExtractionMode>();
    derived.options.remove_json(ChatExtractionMode::NAMESPACE);
    match mode {
        ChatExtractionMode::Native => {
            derived.response_format = Some(ResponseFormat::json_schema(schema).name(name));
        }
        ChatExtractionMode::ForcedTool => {
            derived.tools = vec![
                ToolDefinition::new(EXTRACTION_TOOL, schema)
                    .description("Submit extracted structured data")
                    .strict(true),
            ];
            derived.tool_choice = Some(ToolChoice::Specific {
                name: EXTRACTION_TOOL.to_owned(),
            });
            derived.response_format = None;
            // `parallel_tool_calls` stays the caller's. Dialects whose strict
            // mode breaks under parallel calls (OpenAI fine-tuned models,
            // Azure OpenAI) turn them off in the adapter, which knows the
            // dialect; core does not guess it from the model id.
        }
        ChatExtractionMode::Auto => {
            return Err(ChatExtractError::Unsupported {
                mode,
                provider: provider.to_owned(),
            });
        }
    }
    Ok(derived)
}

/// The request field whose warnings count for `mode`'s enforcement: the
/// one that carries the schema.
fn counted_field(mode: ChatExtractionMode) -> Option<RequestField> {
    match mode {
        ChatExtractionMode::Native => Some(RequestField::ResponseFormat),
        ChatExtractionMode::ForcedTool => Some(RequestField::Tools),
        ChatExtractionMode::Auto => None,
    }
}

/// Whether `warning` is about `field`.
fn is_about(warning: &Warning, field: RequestField) -> bool {
    warning.subject == Subject::Field { field }
}

/// Parse the value from a response the extraction is done with: one that
/// will not be retried.
fn finish_extraction<T: DeserializeOwned>(
    response: ChatResponse,
    mut metadata: ChatExtractionMetadata,
    target: &Target,
    repair: bool,
    reject_undeclared: bool,
) -> Result<ChatExtracted<T>> {
    let parsed = parse_value::<T>(
        &response,
        metadata.mode,
        &target.provider,
        repair,
        Some(&target.schema),
        reject_undeclared,
    );
    match parsed {
        Ok((value, repaired)) => {
            metadata.repaired = repaired;
            Ok(ChatExtracted {
                value,
                response,
                metadata,
            })
        }
        Err(rejection) => {
            if rejection.undeclared {
                metadata.undeclared_key_rejections += 1;
            }
            Err(ChatExtractFailure::new(rejection.error, metadata)
                .with_response(response)
                .into())
        }
    }
}

/// Output that could not be used: why, and for a parse failure, whether
/// the output was JSON at all, which the retry feedback tells apart.
struct Rejection {
    error: ChatExtractError,
    /// The parse failed on JSON syntax (or early end of input), not on the
    /// shape of `T`.
    invalid_json: bool,
    /// The parse failed on a key the schema of `T` does not declare.
    undeclared: bool,
}

impl From<ChatExtractError> for Rejection {
    fn from(error: ChatExtractError) -> Self {
        Self {
            error,
            invalid_json: false,
            undeclared: false,
        }
    }
}

fn is_zero(n: &u32) -> bool {
    *n == 0
}

/// Parse the value, returning whether local repair was needed.
///
/// With the schema of `T`, the stand-in `null`s of optional properties are
/// removed first ([`normalize_strict_nulls`](crate::adapter::schema::normalize_strict_nulls)). The
/// raw output in a parse error stays the text the model sent.
///
/// With repair on, a parse error reports the repair candidate's error when
/// the candidate is valid JSON (it reached `T`, so its error is the one
/// that matters), and the raw text's error otherwise.
fn parse_value<T: DeserializeOwned>(
    response: &ChatResponse,
    mode: ChatExtractionMode,
    provider: &str,
    repair: bool,
    schema: Option<&serde_json::Value>,
    reject_undeclared: bool,
) -> std::result::Result<(T, bool), Rejection> {
    if let Some(refused) = refusal(response, mode, provider) {
        return Err(refused.into());
    }
    if let Some(finish_reason @ (FinishReason::Length | FinishReason::ContextWindow)) =
        &response.finish_reason
    {
        let usage = response.usage.as_ref();
        return Err(ChatExtractError::Truncated {
            mode,
            provider: provider.to_owned(),
            finish_reason: finish_reason.clone(),
            output_tokens: usage.and_then(|usage| usage.output_tokens),
            reasoning_tokens: usage.and_then(|usage| usage.reasoning_tokens),
        }
        .into());
    }
    let raw = match mode {
        ChatExtractionMode::ForcedTool => {
            let calls: Vec<_> = response
                .tool_calls()
                .filter(|call| call.name == EXTRACTION_TOOL)
                .collect();
            match calls.as_slice() {
                [] => {
                    return Err(ChatExtractError::MissingToolCall {
                        provider: provider.to_owned(),
                    }
                    .into());
                }
                [call] => call.arguments.clone(),
                many => {
                    return Err(ChatExtractError::UnexpectedToolCallCount {
                        provider: provider.to_owned(),
                        count: many.len(),
                    }
                    .into());
                }
            }
        }
        _ => response
            .text()
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| ChatExtractError::MissingStructuredText {
                provider: provider.to_owned(),
            })?,
    };
    let mut failure = match crate::schema::from_str_checked(&raw, schema, reject_undeclared) {
        Ok(value) => return Ok((value, false)),
        Err(failure) => failure,
    };
    if repair && let Some(candidate) = repair_candidate(&raw) {
        match crate::schema::from_str_checked(candidate, schema, reject_undeclared) {
            Ok(value) => return Ok((value, true)),
            Err(candidate_failure) if !is_json_syntax_error(&candidate_failure.error) => {
                failure = candidate_failure;
            }
            Err(_) => {}
        }
    }
    let mut message = failure.error.to_string();
    if let Some(reason) = response
        .finish_reason
        .as_ref()
        .filter(|reason| !reason.finished_normally())
    {
        message.push_str(&format!(" (finish reason `{reason}`)"));
    }
    Err(Rejection {
        invalid_json: is_json_syntax_error(&failure.error),
        undeclared: failure.undeclared,
        error: ChatExtractError::Parse {
            mode,
            provider: provider.to_owned(),
            message,
            path: failure.path,
            raw,
        },
    })
}

/// The refusal in `response`, if it is one: a refusal the model wrote, or a
/// `Refusal` or `ContentFilter` finish.
fn refusal(
    response: &ChatResponse,
    mode: ChatExtractionMode,
    provider: &str,
) -> Option<ChatExtractError> {
    let written = model_refusal(response);
    let declined = matches!(
        response.finish_reason,
        Some(FinishReason::Refusal | FinishReason::ContentFilter)
    );
    if written.is_none() && !declined {
        return None;
    }
    let (text, source) = match (written, decline_explanation(response)) {
        (Some(text), _) => (Some(text.to_owned()), RefusalSource::Model),
        (None, Some(explanation)) => (Some(explanation.to_owned()), RefusalSource::Provider),
        (None, None) => (None, RefusalSource::None),
    };
    Some(ChatExtractError::Refused {
        mode,
        provider: provider.to_owned(),
        finish_reason: response.finish_reason.clone(),
        text,
        source,
    })
}

/// The refusal the model wrote in place of the output: its
/// [`Refusal`](crate::Refusal) block.
fn model_refusal(response: &ChatResponse) -> Option<&str> {
    response.first_refusal()
}

/// The finish reason of a [`ChatExtractError::Truncated`] recorded before it
/// had one: the output token limit.
fn output_limit() -> FinishReason {
    FinishReason::Length
}

/// The provider's explanation of its decline, stored by the adapter under
/// its namespace (see
/// [Decline explanations](crate::provider_data#decline-explanations)).
fn decline_explanation(response: &ChatResponse) -> Option<&str> {
    response
        .provider_data
        .namespaces()
        .filter(|(namespace, _)| *namespace != ALLAMA_NAMESPACE)
        .find_map(|(_, entries)| entries.get(DECLINE_EXPLANATION_KEY)?.as_str())
        .filter(|text| !text.trim().is_empty())
}

/// Whether `error` means the text is not valid JSON, as opposed to valid
/// JSON that does not fit the target type.
fn is_json_syntax_error(error: &serde_json::Error) -> bool {
    matches!(
        error.classify(),
        serde_json::error::Category::Syntax | serde_json::error::Category::Eof
    )
}

/// The JSON value inside `raw`, without a markdown fence or surrounding
/// prose. `None` when there is nothing to strip.
fn repair_candidate(raw: &str) -> Option<&str> {
    let trimmed = raw.trim();
    // A fenced block: skip the info string (```json) and cut at the closing
    // fence, ignoring prose after it.
    if let Some((_, body)) = trimmed
        .strip_prefix("```")
        .and_then(|fenced| fenced.split_once('\n'))
    {
        let body = body.find("```").map_or(body, |end| &body[..end]).trim();
        if !body.is_empty() {
            return Some(body);
        }
    }
    let start = trimmed.find(['{', '['])?;
    let end = trimmed.rfind(['}', ']'])?;
    let candidate = trimmed.get(start..=end)?;
    (candidate.len() < trimmed.len()).then_some(candidate)
}

/// Extend `request` with the failed response and feedback, for a retry.
fn ask_again(
    request: &mut ChatRequest,
    response: ChatResponse,
    mode: ChatExtractionMode,
    rejection: &Rejection,
) {
    let calls: Vec<_> = response.tool_calls().cloned().collect();
    let has_content = !response.content.is_empty();
    if has_content {
        request.push_message(response.into_assistant_message());
    }
    let problem = match &rejection.error {
        ChatExtractError::Parse { message, .. } if rejection.invalid_json => {
            format!("it was not valid JSON: {message}")
        }
        ChatExtractError::Parse {
            message,
            path: Some(path),
            ..
        } => format!("it did not match the schema at `{path}`: {message}"),
        ChatExtractError::Parse { message, .. } => {
            format!("it did not match the schema: {message}")
        }
        ChatExtractError::MissingStructuredText { .. } => "it contained no text".to_owned(),
        ChatExtractError::MissingToolCall { .. } => {
            format!("it did not call the `{EXTRACTION_TOOL}` tool")
        }
        ChatExtractError::UnexpectedToolCallCount { count, .. } => {
            format!("it called the `{EXTRACTION_TOOL}` tool {count} times instead of once")
        }
        other => other.to_string(),
    };
    let forced = mode == ChatExtractionMode::ForcedTool;
    // Every replayed tool call needs a result before the next turn, in any
    // mode (a native-mode request may carry the caller's tools). An ordinary
    // result, not a failed one: the text already says the call failed, and
    // an endpoint with no tool error flag would otherwise warn about the
    // flag (the bundled adapters warn `approximated.tool_result_error`), or,
    // in an adapter that refuses it, fail the turn under the strict policy.
    if has_content {
        for call in &calls {
            let feedback = if forced {
                format!(
                    "The previous call could not be used: {problem}. Call `{EXTRACTION_TOOL}` once with arguments that match its schema."
                )
            } else {
                "This tool call was not run: a structured reply is required.".to_owned()
            };
            request.push_tool_result(call, feedback);
        }
    }
    if forced && has_content && !calls.is_empty() {
        return;
    }
    request.push_message(Message::user(if forced {
        format!(
            "The previous reply could not be used: {problem}. Call `{EXTRACTION_TOOL}` once with the extracted data."
        )
    } else {
        format!(
            "The previous reply could not be used: {problem}. Reply again with only the JSON value, matching the schema exactly."
        )
    }));
}

fn schema_name<T>(schema: &serde_json::Value) -> String {
    let raw = schema
        .get("title")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| {
            std::any::type_name::<T>()
                .rsplit("::")
                .next()
                .unwrap_or("structured_output")
                .to_owned()
        });
    let mut name = String::with_capacity(raw.len());
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() {
            name.push(ch.to_ascii_lowercase());
        } else if !name.ends_with('_') && !name.is_empty() {
            name.push('_');
        }
    }
    let trimmed = name.trim_matches('_');
    if trimmed.is_empty() {
        "structured_output".into()
    } else {
        trimmed.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::{ChatResponseBuilder, MockProvider, MockStreamEvent, MockStreamingProvider};
    use crate::{CapabilitySupport, UserContent};
    use futures_util::StreamExt;
    use serde_json::json;

    #[derive(Debug, PartialEq, Deserialize, JsonSchema)]
    struct Person {
        name: String,
        age: u8,
    }

    fn request() -> ChatRequest {
        ChatRequest::new("m").user("Alice is 31")
    }

    fn alice() -> Person {
        Person {
            name: "Alice".into(),
            age: 31,
        }
    }

    /// The mock the docs recommend for handler tests answers an extraction
    /// in every mode.
    #[tokio::test]
    async fn extract_works_against_the_test_provider_in_every_mode() {
        #[derive(Debug, PartialEq, Deserialize, JsonSchema)]
        #[serde(rename_all = "snake_case")]
        enum Category {
            Billing,
            Bug,
        }

        #[derive(Debug, PartialEq, Deserialize, JsonSchema)]
        struct Triage {
            category: Category,
            urgent: bool,
            age: u8,
        }

        let provider = crate::mock::TestProvider::new();
        for mode in [
            ChatExtractionMode::Auto,
            ChatExtractionMode::Native,
            ChatExtractionMode::ForcedTool,
        ] {
            let request = ChatRequest::new("m")
                .user("My invoice is wrong")
                .with_option(mode);
            let extracted = provider
                .extract::<Triage>(&request)
                .await
                .unwrap_or_else(|error| panic!("{mode:?}: {error}"));
            assert_eq!(
                extracted.value,
                Triage {
                    category: Category::Billing,
                    urgent: true,
                    age: 0,
                },
                "{mode:?}"
            );
            let expected = match mode {
                ChatExtractionMode::ForcedTool => ChatExtractionMode::ForcedTool,
                _ => ChatExtractionMode::Native,
            };
            assert_eq!(extracted.metadata.mode, expected);
        }

        let mut stream = provider
            .extract_stream::<Triage>(&ChatRequest::new("m").user("My invoice is wrong"))
            .await
            .unwrap();
        while stream.next().await.is_some() {}
        assert_eq!(stream.finish().unwrap().value.category, Category::Billing);
    }

    #[tokio::test]
    async fn native_mode_uses_json_schema() {
        let provider = MockProvider::with_text(r#"{"name":"Alice","age":31}"#)
            .with_chat_capability(
                ChatCapability::StructuredOutput,
                CapabilitySupport::Supported,
            );
        let extracted = provider.extract::<Person>(&request()).await.unwrap();
        assert_eq!(extracted.value, alice());
        assert_eq!(
            extracted.metadata,
            ChatExtractionMetadata {
                mode: ChatExtractionMode::Native,
                passes: 1,
                repaired: false,
                enforcement: SchemaEnforcement::Exact,
                degradations: Vec::new(),
                usage: None,
                auto_fallback: None,
                undeclared_key_rejections: 0,
            }
        );
        let sent = provider.last_request().unwrap();
        let Some(ResponseFormat::JsonSchema { name, strict, .. }) = sent.response_format else {
            panic!("expected json schema");
        };
        assert_eq!(name.as_deref(), Some("person"));
        assert_eq!(strict, Some(true));
    }

    /// Tier 2: verified tool answers beat a guess at native output.
    #[tokio::test]
    async fn auto_picks_forced_tool_without_structured_output() {
        let provider =
            MockProvider::with_tool_call("c", EXTRACTION_TOOL, json!({"name": "Alice", "age": 31}))
                .with_supported_chat_capabilities([
                    ChatCapability::ToolCalls,
                    ChatCapability::ForcedToolChoice,
                ]);
        let extracted = provider.extract::<Person>(&request()).await.unwrap();
        assert_eq!(extracted.metadata.mode, ChatExtractionMode::ForcedTool);
        let sent = provider.last_request().unwrap();
        assert_eq!(
            sent.tool_choice,
            Some(ToolChoice::Specific {
                name: EXTRACTION_TOOL.into()
            })
        );
        assert!(sent.option::<ChatExtractionMode>().is_none());
    }

    /// The refusal an adapter returns, through `Degradations`, for a
    /// response format it cannot send.
    fn refused(field: RequestField) -> Error {
        crate::Degradations::new("demo", crate::UnsupportedPolicy::Warn)
            .unsupported(field, "map-typed properties cannot be expressed")
    }

    fn refusal_record() -> AutoFallback {
        AutoFallback::new(
            RequestField::ResponseFormat,
            "cannot send `response_format`: map-typed properties cannot be expressed",
        )
    }

    fn alice_call() -> serde_json::Value {
        json!({"name": "Alice", "age": 31})
    }

    const NATIVE_AND_TOOLS: [ChatCapability; 2] =
        [ChatCapability::StructuredOutput, ChatCapability::ToolCalls];

    /// When `Auto` only guessed native output from an `Unknown` answer, a
    /// provider's own refusal of the response format (a 4xx, nothing to
    /// bill) also falls back; after a verified `Supported` it does not.
    #[tokio::test]
    async fn a_guessed_native_mode_falls_back_on_the_providers_refusal() {
        let upstream = || {
            Error::unsupported_by_provider(
                Some(crate::Subject::Field {
                    field: RequestField::ResponseFormat,
                }),
                "HTTP 400: the model does not support `response_format`",
            )
            .authored()
        };
        let unknown = MockProvider::build(|b| {
            b.error(upstream())
                .tool_call("c", EXTRACTION_TOOL, alice_call())
        })
        .with_chat_capability(ChatCapability::StructuredOutput, CapabilitySupport::Unknown)
        .with_chat_capability(ChatCapability::ToolCalls, CapabilitySupport::Unknown)
        .with_chat_capability(ChatCapability::ForcedToolChoice, CapabilitySupport::Unknown);
        let extracted = unknown.extract::<Person>(&request()).await.unwrap();
        assert_eq!(extracted.metadata.mode, ChatExtractionMode::ForcedTool);
        let fallback = extracted.metadata.auto_fallback.unwrap();
        assert_eq!(fallback.stage, crate::UnsupportedStage::Provider);
        assert_eq!(fallback.field, RequestField::ResponseFormat);

        let verified = MockProvider::build(|b| {
            b.error(upstream())
                .tool_call("c", EXTRACTION_TOOL, alice_call())
        })
        .with_supported_chat_capabilities(NATIVE_AND_TOOLS);
        let err = verified.extract::<Person>(&request()).await.unwrap_err();
        assert_eq!(
            err.unsupported_stage(),
            Some(crate::UnsupportedStage::Provider)
        );
        assert_eq!(verified.call_count(), 1);
    }

    #[tokio::test]
    async fn auto_retries_a_refused_response_format_in_forced_tool_mode() {
        let provider = MockProvider::build(|b| {
            b.error(refused(RequestField::ResponseFormat)).tool_call(
                "c",
                EXTRACTION_TOOL,
                alice_call(),
            )
        })
        .with_supported_chat_capabilities(NATIVE_AND_TOOLS);
        let extracted = provider.extract::<Person>(&request()).await.unwrap();
        assert_eq!(extracted.value, alice());
        assert_eq!(
            extracted.metadata,
            ChatExtractionMetadata {
                mode: ChatExtractionMode::ForcedTool,
                passes: 1,
                repaired: false,
                // The switch adds no degradation; the forced-tool pass
                // reported none.
                enforcement: SchemaEnforcement::Exact,
                degradations: Vec::new(),
                usage: None,
                auto_fallback: Some(refusal_record()),
                undeclared_key_rejections: 0,
            }
        );
        let [native, forced] = provider.requests().try_into().unwrap();
        assert!(native.response_format.is_some());
        assert!(native.tools.is_empty());
        assert!(forced.response_format.is_none());
        assert_eq!(
            forced.tool_choice,
            Some(ToolChoice::Specific {
                name: EXTRACTION_TOOL.into()
            })
        );
    }

    #[tokio::test]
    async fn only_a_refused_response_format_falls_back_and_only_from_auto() {
        let cases = [
            // An explicit mode is the caller's choice.
            (
                request().with_option(ChatExtractionMode::Native),
                refused(RequestField::ResponseFormat),
            ),
            // A refusal of another field, or one that names none (an
            // upstream reply mapped to `Unsupported`), is not about native
            // output.
            (request(), refused(RequestField::Temperature)),
            (request(), Error::unsupported("HTTP 400")),
            (request(), rate_limited()),
            // The provider refused the response format it received: the
            // call was sent, so Auto does not double-bill by retrying it.
            (
                request(),
                Error::unsupported_by_provider(
                    Some(crate::Subject::Field {
                        field: RequestField::ResponseFormat,
                    }),
                    "HTTP 400",
                ),
            ),
        ];
        for (request, error) in cases {
            let provider = MockProvider::build(|b| {
                b.error(error.clone())
                    .tool_call("c", EXTRACTION_TOOL, alice_call())
            })
            .with_supported_chat_capabilities(NATIVE_AND_TOOLS);
            let err = provider.extract::<Person>(&request).await.unwrap_err();
            assert_eq!(err.kind_name(), error.kind_name(), "{err:?}");
            assert_eq!(err.unsupported_field(), error.unsupported_field());
            let record = ChatExtractionMetadata::from_error(&err).unwrap();
            assert_eq!(record.mode, ChatExtractionMode::Native);
            assert_eq!(record.auto_fallback, None);
            assert_eq!(provider.call_count(), 1, "{error:?}");
        }
    }

    #[tokio::test]
    async fn a_refusal_after_a_pass_does_not_switch_modes() {
        let provider = MockProvider::build(|b| {
            b.text("not json")
                .error(refused(RequestField::ResponseFormat))
        })
        .with_supported_chat_capabilities(NATIVE_AND_TOOLS);
        let err = ChatExtractor::new()
            .retries(1)
            .extract::<_, Person>(&provider, &request())
            .await
            .unwrap_err();
        assert_eq!(err.unsupported_field(), Some(RequestField::ResponseFormat));
        let record = ChatExtractionMetadata::from_error(&err).unwrap();
        assert_eq!(
            (record.mode, record.passes),
            (ChatExtractionMode::Native, 1)
        );
        assert_eq!(record.auto_fallback, None);
    }

    #[tokio::test]
    async fn a_refusal_ending_a_fallback_chain_falls_back() {
        let chained = refused(RequestField::ResponseFormat).with_fallback_from(rate_limited());
        let provider = MockProvider::build(|b| {
            b.error(chained)
                .tool_call("c", EXTRACTION_TOOL, alice_call())
        })
        .with_supported_chat_capabilities(NATIVE_AND_TOOLS);
        let extracted = provider.extract::<Person>(&request()).await.unwrap();
        assert_eq!(extracted.metadata.mode, ChatExtractionMode::ForcedTool);
        assert_eq!(extracted.metadata.auto_fallback, Some(refusal_record()));
    }

    #[tokio::test]
    async fn a_failed_forced_tool_attempt_reports_the_native_refusal_first() {
        let provider = MockProvider::build(|b| {
            b.error(refused(RequestField::ResponseFormat))
                .error(refused(RequestField::Tools))
        })
        .with_supported_chat_capabilities(NATIVE_AND_TOOLS);
        let err = provider.extract::<Person>(&request()).await.unwrap_err();
        // The native refusal is the primary cause, in terms of the type.
        assert_eq!(err.unsupported_field(), Some(RequestField::ResponseFormat));
        assert_eq!(
            err.message(),
            "`Person` cannot be extracted from `m` with its output constrained: \
             native output was refused (cannot send `response_format`: map-typed properties cannot be expressed), \
             and forced-tool mode was refused too (cannot send `tools`: map-typed properties cannot be expressed)"
        );
        assert!(!err.message().contains(EXTRACTION_TOOL));
        let record = ChatExtractionMetadata::from_error(&err).unwrap();
        assert_eq!(
            (record.mode, record.passes),
            (ChatExtractionMode::ForcedTool, 0)
        );
        assert_eq!(record.auto_fallback, Some(refusal_record()));
        assert_eq!(provider.call_count(), 2);
    }

    /// A request with tools of its own is refused
    /// the same way in every mode, before anything is sent, so the same
    /// code does not work on one provider and fail on another.
    #[tokio::test]
    async fn request_tools_conflict_in_every_mode() {
        let request = request().tools([ToolDefinition::new("other", json!({"type": "object"}))]);
        for mode in [
            None,
            Some(ChatExtractionMode::Native),
            Some(ChatExtractionMode::ForcedTool),
        ] {
            for capabilities in [&NATIVE_AND_TOOLS[..], &[ChatCapability::ToolCalls][..]] {
                let provider = MockProvider::with_text(r#"{"name":"Alice","age":31}"#)
                    .with_supported_chat_capabilities(capabilities.iter().copied());
                let request = match mode {
                    Some(mode) => request.clone().with_option(mode),
                    None => request.clone(),
                };
                let failure = failure(provider.extract::<Person>(&request).await.unwrap_err());
                assert!(
                    matches!(failure.error(), ChatExtractError::ToolConflict { .. }),
                    "{mode:?}: {}",
                    failure.error()
                );
                assert!(
                    failure
                        .error()
                        .to_string()
                        .contains("start a fresh request")
                );
                assert_eq!(failure.metadata().passes, 0);
                assert_eq!(provider.call_count(), 0, "{mode:?}");
            }
        }
    }

    #[tokio::test]
    async fn a_refusal_without_authored_text_keeps_no_message() {
        // Only `Degradations` sets the field, and its text is authored; a
        // deserialized error may not be.
        let mut json = serde_json::to_value(refused(RequestField::ResponseFormat)).unwrap();
        json["data"]["context"]
            .as_object_mut()
            .unwrap()
            .remove("message_origin");
        let unauthored: Error = serde_json::from_value(json).unwrap();
        assert_eq!(unauthored.message_origin(), crate::MessageOrigin::Unknown);
        let provider = MockProvider::build(|b| {
            b.error(unauthored)
                .tool_call("c", EXTRACTION_TOOL, alice_call())
        })
        .with_supported_chat_capabilities(NATIVE_AND_TOOLS);
        let extracted = provider.extract::<Person>(&request()).await.unwrap();
        assert_eq!(
            extracted.metadata.auto_fallback,
            Some(AutoFallback::new(RequestField::ResponseFormat, ""))
        );
    }

    async fn drain<S, T>(stream: &mut ChatExtractionStream<S, T>) -> usize
    where
        S: Stream<Item = Result<StreamEvent>>,
    {
        let mut events = 0;
        while let Some(event) = stream.next().await {
            event.unwrap();
            events += 1;
        }
        events
    }

    /// A provider whose stream is its items as given, with no
    /// `finish_stream` to keep it to the stream rules.
    struct RawStream(std::sync::Mutex<Option<Vec<Result<StreamEvent>>>>);

    impl crate::ProviderIdentity for RawStream {
        fn provider_name(&self) -> &str {
            "raw"
        }

        fn reads_namespace(&self, namespace: &str) -> bool {
            namespace == "raw"
        }
    }

    impl ChatProvider for RawStream {
        type Stream = futures_util::stream::Iter<std::vec::IntoIter<Result<StreamEvent>>>;

        async fn chat(&self, _: &ChatRequest) -> Result<ChatResponse> {
            unreachable!("streams only")
        }

        async fn chat_stream(&self, _: &ChatRequest) -> Result<Self::Stream> {
            let items = self.0.lock().unwrap().take().unwrap_or_default();
            Ok(futures_util::stream::iter(items))
        }

        fn chat_capability(&self, _: &str, _: ChatCapability) -> CapabilitySupport {
            CapabilitySupport::Supported
        }
    }

    #[tokio::test]
    async fn a_streamed_extraction_ends_after_the_first_error() {
        let provider = RawStream(std::sync::Mutex::new(Some(vec![
            Ok(StreamEvent::response_start(None, None)),
            Err(Error::stream_decode("broken")),
            Ok(StreamEvent::text_delta(0, "after")),
        ])));
        let mut stream = provider.extract_stream::<Person>(&request()).await.unwrap();
        assert!(matches!(
            stream.next().await,
            Some(Ok(StreamEvent::ResponseStart { .. }))
        ));
        assert!(matches!(stream.next().await, Some(Err(_))));
        assert!(stream.next().await.is_none());
        assert!(stream.finish().is_err());
    }

    #[tokio::test]
    async fn streamed_auto_falls_back_on_a_setup_refusal() {
        let provider = MockStreamingProvider::build(|b| {
            b.setup_error(refused(RequestField::ResponseFormat))
                .tool_call("c", EXTRACTION_TOOL, alice_call())
        })
        .with_chat_capability(
            ChatCapability::StructuredOutput,
            CapabilitySupport::Supported,
        );
        let mut stream = provider.extract_stream::<Person>(&request()).await.unwrap();
        assert_eq!(stream.mode(), ChatExtractionMode::ForcedTool);
        drain(&mut stream).await;
        let extracted = stream.finish().unwrap();
        assert_eq!(extracted.value, alice());
        assert_eq!(extracted.metadata.mode, ChatExtractionMode::ForcedTool);
        assert_eq!(extracted.metadata.passes, 1);
        assert_eq!(extracted.metadata.auto_fallback, Some(refusal_record()));
        assert_eq!(provider.call_count(), 2);
        assert!(provider.requests()[1].tool_choice.is_some());
    }

    #[tokio::test]
    async fn streamed_auto_falls_back_on_a_refusal_as_the_first_item() {
        let provider = MockStreamingProvider::new([
            vec![MockStreamEvent::from(refused(RequestField::ResponseFormat))],
            MockStreamEvent::tool_call_response("c", EXTRACTION_TOOL, alice_call()),
        ])
        .with_chat_capability(
            ChatCapability::StructuredOutput,
            CapabilitySupport::Supported,
        );
        let mut stream = provider.extract_stream::<Person>(&request()).await.unwrap();
        assert_eq!(stream.mode(), ChatExtractionMode::ForcedTool);
        drain(&mut stream).await;
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(stream.partial_output()).unwrap(),
            alice_call()
        );
        let extracted = stream.finish().unwrap();
        assert_eq!(extracted.value, alice());
        assert_eq!(extracted.metadata.auto_fallback, Some(refusal_record()));
        assert_eq!(provider.call_count(), 2);
    }

    #[tokio::test]
    async fn streamed_auto_yields_the_first_item_it_read() {
        let provider = MockStreamingProvider::with_text(r#"{"name":"Alice","age":31}"#)
            .with_chat_capability(
                ChatCapability::StructuredOutput,
                CapabilitySupport::Supported,
            );
        let mut stream = provider.extract_stream::<Person>(&request()).await.unwrap();
        let events = drain(&mut stream).await;
        let expected = MockStreamEvent::text_response(r#"{"name":"Alice","age":31}"#).len();
        assert_eq!(events, expected);
        let extracted = stream.finish().unwrap();
        assert_eq!(extracted.value, alice());
        assert_eq!(extracted.metadata.mode, ChatExtractionMode::Native);
        assert_eq!(extracted.metadata.auto_fallback, None);

        // A first item that is not a refusal of the response format is
        // yielded as the stream's error.
        let provider = MockStreamingProvider::with_stream([MockStreamEvent::from(refused(
            RequestField::Tools,
        ))])
        .with_chat_capability(
            ChatCapability::StructuredOutput,
            CapabilitySupport::Supported,
        );
        let mut stream = provider.extract_stream::<Person>(&request()).await.unwrap();
        let first = stream.next().await.unwrap().unwrap_err();
        assert_eq!(first.unsupported_field(), Some(RequestField::Tools));
        assert!(stream.next().await.is_none());
        let err = stream.finish().unwrap_err();
        assert_eq!(err.unsupported_field(), Some(RequestField::Tools));
        assert_eq!(provider.call_count(), 1);

        // An empty stream was cut short: it fails where it would have.
        let provider = MockStreamingProvider::with_stream(Vec::<MockStreamEvent>::new())
            .with_chat_capability(
                ChatCapability::StructuredOutput,
                CapabilitySupport::Supported,
            );
        let mut stream = provider.extract_stream::<Person>(&request()).await.unwrap();
        let first = stream.next().await.unwrap().unwrap_err();
        assert!(matches!(
            first,
            Error::Stream {
                kind: crate::StreamErrorKind::Incomplete,
                ..
            }
        ));
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn streamed_failed_fallback_returns_the_forced_tool_error() {
        let provider = MockStreamingProvider::build(|b| {
            b.setup_error(refused(RequestField::ResponseFormat))
                .setup_error(rate_limited())
        })
        .with_chat_capability(
            ChatCapability::StructuredOutput,
            CapabilitySupport::Supported,
        );
        let Err(err) = provider.extract_stream::<Person>(&request()).await else {
            panic!("expected the forced-tool attempt's error");
        };
        assert!(matches!(err, Error::RateLimited { .. }), "{err:?}");
        let record = ChatExtractionMetadata::from_error(&err).unwrap();
        assert_eq!(record.mode, ChatExtractionMode::ForcedTool);
        assert_eq!(record.auto_fallback, Some(refusal_record()));

        // An explicit native stream never switches.
        let provider =
            MockStreamingProvider::build(|b| b.setup_error(refused(RequestField::ResponseFormat)));
        let request = request().with_option(ChatExtractionMode::Native);
        let Err(err) = provider.extract_stream::<Person>(&request).await else {
            panic!("expected the refusal");
        };
        assert_eq!(err.unsupported_field(), Some(RequestField::ResponseFormat));
        assert_eq!(provider.call_count(), 1);
    }

    #[test]
    fn the_auto_fallback_record_round_trips() {
        let mut metadata = ChatExtractionMetadata::new(ChatExtractionMode::Native);
        metadata.fall_back(refusal_record());
        let value = serde_json::to_value(&metadata).unwrap();
        assert_eq!(value["mode"], "forced_tool");
        assert_eq!(value["auto_fallback"]["field"], "response_format");
        assert_eq!(
            serde_json::from_value::<ChatExtractionMetadata>(value).unwrap(),
            metadata
        );
        let plain =
            serde_json::to_value(ChatExtractionMetadata::new(ChatExtractionMode::Native)).unwrap();
        assert!(plain.get("auto_fallback").is_none(), "{plain}");
    }

    /// Tier 5: the answers rule out both modes.
    #[tokio::test]
    async fn auto_skips_forced_tool_when_the_model_rejects_it() {
        let provider =
            MockProvider::with_tool_call("c", EXTRACTION_TOOL, json!({"name": "Alice", "age": 31}))
                .with_chat_capability(ChatCapability::ToolCalls, CapabilitySupport::Supported)
                .with_chat_capability(
                    ChatCapability::ForcedToolChoice,
                    CapabilitySupport::Unsupported,
                )
                .with_chat_capability(
                    ChatCapability::StructuredOutput,
                    CapabilitySupport::Unsupported,
                );
        let err = provider.extract::<Person>(&request()).await.unwrap_err();
        let Error::Extract { failure, .. } = err else {
            panic!("expected extract error");
        };
        assert!(
            matches!(failure.error(), ChatExtractError::Unsupported { .. }),
            "{failure:?}"
        );
        assert!(failure.response().is_none());
        assert_eq!(failure.metadata().passes, 0);
        assert_eq!(provider.call_count(), 0);
    }

    #[tokio::test]
    async fn explicit_mode_option_and_parse_errors() {
        let provider = MockProvider::with_text("not json");
        let request = request().with_option(ChatExtractionMode::Native);
        let err = provider.extract::<Person>(&request).await.unwrap_err();
        let Error::Extract { failure, .. } = err else {
            panic!("expected extract error");
        };
        assert_eq!(failure.error().raw_output(), Some("not json"));
    }

    /// Tier 3: with nothing known, native output is a guess, never a
    /// failure.
    #[tokio::test]
    async fn unknown_capabilities_guess_native_output() {
        let provider = MockProvider::with_text(r#"{"name":"Alice","age":31}"#);
        let extracted = provider.extract::<Person>(&request()).await.unwrap();
        assert_eq!(extracted.metadata.mode, ChatExtractionMode::Native);
        assert!(provider.last_request().unwrap().response_format.is_some());

        // Even when the tools are verified, forced choice is not.
        let provider = MockProvider::with_text(r#"{"name":"Alice","age":31}"#)
            .with_chat_capability(ChatCapability::ToolCalls, CapabilitySupport::Supported);
        let extracted = provider.extract::<Person>(&request()).await.unwrap();
        assert_eq!(extracted.metadata.mode, ChatExtractionMode::Native);
    }

    /// Tier 4: without native output, unknown tool answers are a guess too.
    #[tokio::test]
    async fn without_native_output_unknown_tools_are_guessed() {
        let provider =
            MockProvider::with_tool_call("c", EXTRACTION_TOOL, json!({"name": "Alice", "age": 31}))
                .with_chat_capability(
                    ChatCapability::StructuredOutput,
                    CapabilitySupport::Unsupported,
                );
        let extracted = provider.extract::<Person>(&request()).await.unwrap();
        assert_eq!(extracted.metadata.mode, ChatExtractionMode::ForcedTool);

        let provider = MockProvider::empty()
            .with_chat_capability(
                ChatCapability::StructuredOutput,
                CapabilitySupport::Unsupported,
            )
            .with_chat_capability(ChatCapability::ToolCalls, CapabilitySupport::Unsupported);
        let err = provider.extract::<Person>(&request()).await.unwrap_err();
        assert!(
            matches!(err, Error::Extract { ref failure, .. } if matches!(failure.error(), ChatExtractError::Unsupported { .. })),
            "{err:?}"
        );
        assert!(err.to_string().contains("capability resolver"), "{err}");
        assert_eq!(provider.call_count(), 0);
    }

    #[tokio::test]
    async fn forced_tool_conflicts_with_request_tools() {
        let provider = MockProvider::empty();
        let request = request()
            .with_option(ChatExtractionMode::ForcedTool)
            .tools([ToolDefinition::new("other", json!({"type": "object"}))]);
        assert!(provider.extract::<Person>(&request).await.is_err());
    }

    #[tokio::test]
    async fn extraction_through_wrappers() {
        let provider = crate::wrappers::RetryingProvider::new(
            MockProvider::with_response(
                ChatResponseBuilder::new()
                    .text(r#"{"name":"Bob","age":4}"#)
                    .build(),
            ),
            crate::wrappers::RetryPolicy::default(),
        );
        let request = ChatRequest::new("m")
            .user("Bob is 4")
            .with_option(ChatExtractionMode::Native);
        let extracted: ChatExtracted<Person> = ChatExtractor::new()
            .extract(&provider, &request)
            .await
            .unwrap();
        assert_eq!(extracted.value.name, "Bob");
    }

    #[test]
    fn repair_candidates() {
        assert_eq!(
            repair_candidate("```json\n{\"a\": 1}\n```"),
            Some("{\"a\": 1}")
        );
        assert_eq!(repair_candidate("```\n[1]\n```\n"), Some("[1]"));
        assert_eq!(
            repair_candidate("Sure! Here it is: {\"a\": 1}. Anything else?"),
            Some("{\"a\": 1}")
        );
        assert_eq!(repair_candidate("{\"a\": 1}"), None);
        assert_eq!(repair_candidate("no json here"), None);
        assert_eq!(repair_candidate("} backwards {"), None);
        assert_eq!(
            repair_candidate("```json\n{\"a\": 1}\n```\nHope this helps!"),
            Some("{\"a\": 1}")
        );
        assert_eq!(repair_candidate("```{\"a\": 1}```"), Some("{\"a\": 1}"));
    }

    #[tokio::test]
    async fn repair_is_opt_in_and_reported() {
        let fenced = "```json\n{\"name\":\"Alice\",\"age\":31}\n```";
        let request = request().with_option(ChatExtractionMode::Native);

        let strict = MockProvider::with_text(fenced);
        assert!(strict.extract::<Person>(&request).await.is_err());

        let provider = MockProvider::with_text(fenced);
        let extracted = ChatExtractor::new()
            .repair(true)
            .extract::<_, Person>(&provider, &request)
            .await
            .unwrap();
        assert_eq!(extracted.value, alice());
        assert!(extracted.metadata.repaired);
        assert_eq!(provider.call_count(), 1);
    }

    #[tokio::test]
    async fn native_retries_send_the_failure_back() {
        let provider = MockProvider::build(|b| {
            b.text(r#"{"name":"Alice"}"#)
                .text(r#"{"name":"Alice","age":31}"#)
        });
        let request = request().with_option(ChatExtractionMode::Native);
        let extracted = ChatExtractor::new()
            .retries(2)
            .extract::<_, Person>(&provider, &request)
            .await
            .unwrap();
        assert_eq!(extracted.value, alice());
        assert_eq!(extracted.metadata.passes, 2);

        let retry = provider.last_request().unwrap();
        assert_eq!(retry.messages.len(), 3);
        assert!(retry.messages[1].as_assistant().is_some());
        let UserContent::Text(feedback) = &retry.messages[2].as_user().unwrap().content else {
            panic!("expected text feedback");
        };
        assert!(feedback.contains("missing field `age`"), "{feedback}");
    }

    #[tokio::test]
    async fn native_retries_answer_tool_calls_before_asking_again() {
        let provider = MockProvider::build(|b| {
            b.tool_call("c1", "lookup", json!({"q": "Alice"}))
                .text(r#"{"name":"Alice","age":31}"#)
        });
        // A model may call a tool it was not offered.
        let request = request().with_option(ChatExtractionMode::Native);
        let extracted = ChatExtractor::new()
            .retries(1)
            .extract::<_, Person>(&provider, &request)
            .await
            .unwrap();
        assert_eq!(extracted.value, alice());

        let retry = provider.last_request().unwrap();
        assert_eq!(retry.messages.len(), 4);
        let result = retry.messages[2].as_tool().unwrap();
        assert_eq!(result.tool_call_id, "c1");
        // An ordinary result whose text says the call was not run.
        assert!(!result.is_error);
        let feedback = result.content.to_text().unwrap_or_default();
        assert!(feedback.contains("was not run"), "{feedback}");
        assert!(retry.messages[3].as_user().is_some());
    }

    #[tokio::test]
    async fn forced_tool_retries_answer_the_failed_call() {
        let provider = MockProvider::build(|b| {
            b.tool_call("c1", EXTRACTION_TOOL, json!({"name": "Alice"}))
                .tool_call("c2", EXTRACTION_TOOL, json!({"name": "Alice", "age": 31}))
        });
        let request = request().with_option(ChatExtractionMode::ForcedTool);
        let extracted = ChatExtractor::new()
            .retries(1)
            .extract::<_, Person>(&provider, &request)
            .await
            .unwrap();
        assert_eq!(extracted.value, alice());
        assert_eq!(extracted.metadata.passes, 2);

        let retry = provider.last_request().unwrap();
        let result = retry.messages[2].as_tool().unwrap();
        assert_eq!(result.tool_call_id, "c1");
        assert!(!result.is_error);
        let feedback = result.content.to_text().unwrap_or_default();
        assert!(feedback.contains("could not be used"), "{feedback}");
        assert!(feedback.contains("missing field `age`"), "{feedback}");
        assert_eq!(
            retry.tool_choice,
            Some(ToolChoice::Specific {
                name: EXTRACTION_TOOL.into()
            })
        );
    }

    /// The retry turn answered the failed call with a
    /// failed tool result, which an endpoint with no tool error flag refuses
    /// under the strict policy, so a strict extraction tripped over its own
    /// retry.
    #[tokio::test]
    async fn a_strict_retry_sends_on_an_endpoint_without_a_tool_error_flag() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        use crate::mock::FunctionProvider;

        let calls = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&calls);
        // Refuses a failed tool result under the strict policy, as an
        // adapter for such an endpoint does.
        let provider = FunctionProvider::new(move |request: &ChatRequest| {
            if counter.fetch_add(1, Ordering::SeqCst) == 0 {
                return Ok(ChatResponseBuilder::new()
                    .tool_call("c1", EXTRACTION_TOOL, json!({"name": "Alice"}))
                    .build());
            }
            if request
                .messages
                .iter()
                .any(|m| m.as_tool().is_some_and(|t| t.is_error))
            {
                return Err(Error::unsupported("no tool error flag"));
            }
            Ok(ChatResponseBuilder::new()
                .tool_call("c2", EXTRACTION_TOOL, alice_call())
                .build())
        })
        .with_chat_capability(ChatCapability::ToolCalls, CapabilitySupport::Supported)
        .with_chat_capability(
            ChatCapability::ForcedToolChoice,
            CapabilitySupport::Supported,
        );
        let request = request()
            .with_option(ChatExtractionMode::ForcedTool)
            .unsupported_policy(crate::UnsupportedPolicy::Error);
        let extracted = ChatExtractor::new()
            .retries(1)
            .extract::<_, Person>(&provider, &request)
            .await
            .unwrap();
        assert_eq!(extracted.value, alice());
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn retries_are_bounded_and_skip_provider_errors() {
        let provider = MockProvider::build(|b| b.text("nope").text("still nope"));
        let request = request().with_option(ChatExtractionMode::Native);
        let err = ChatExtractor::new()
            .retries(1)
            .extract::<_, Person>(&provider, &request)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Extract { .. }));
        assert_eq!(provider.call_count(), 2);

        let failing = MockProvider::build(|b| b.error(Error::auth("bad key")));
        let err = ChatExtractor::new()
            .retries(3)
            .extract::<_, Person>(&failing, &request)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Auth { .. }));
        assert_eq!(failing.call_count(), 1);
    }

    #[derive(Debug, PartialEq, Deserialize, JsonSchema)]
    enum Currency {
        Usd,
        Eur,
    }

    #[derive(Debug, PartialEq, Deserialize, JsonSchema)]
    struct Price {
        currency: Currency,
    }

    /// The retry feedback the model received: the last user message.
    fn retry_feedback(provider: &MockProvider) -> String {
        let retry = provider.last_request().unwrap();
        let UserContent::Text(feedback) =
            &retry.messages.last().unwrap().as_user().unwrap().content
        else {
            panic!("expected text feedback");
        };
        feedback.clone()
    }

    #[tokio::test]
    async fn repair_reports_the_candidate_error_when_it_is_json() {
        let raw = "Here:\n{\"currency\":\"Gbp\"}";
        let provider = MockProvider::build(|b| b.text(raw).text(raw));
        let request = request().with_option(ChatExtractionMode::Native);
        let err = ChatExtractor::new()
            .repair(true)
            .retries(1)
            .extract::<_, Price>(&provider, &request)
            .await
            .unwrap_err();
        let failure = failure(err);
        let ChatExtractError::Parse {
            message,
            raw: reported,
            ..
        } = failure.error()
        else {
            panic!("expected a parse failure, got {failure:?}");
        };
        assert!(message.contains("unknown variant `Gbp`"), "{message}");
        assert_eq!(reported, raw);

        let feedback = retry_feedback(&provider);
        assert!(
            feedback.contains("it did not match the schema at `currency`: unknown variant `Gbp`"),
            "{feedback}"
        );
    }

    #[tokio::test]
    async fn repair_keeps_the_raw_error_when_the_candidate_is_not_json() {
        let raw = "Here: {\"currency\": Usd}";
        let provider = MockProvider::with_text(raw);
        let request = request().with_option(ChatExtractionMode::Native);
        let err = ChatExtractor::new()
            .repair(true)
            .extract::<_, Price>(&provider, &request)
            .await
            .unwrap_err();
        let failure = failure(err);
        let ChatExtractError::Parse {
            message,
            raw: reported,
            ..
        } = failure.error()
        else {
            panic!("expected a parse failure, got {failure:?}");
        };
        let raw_error = serde_json::from_str::<serde_json::Value>(raw)
            .unwrap_err()
            .to_string();
        assert_eq!(message, &raw_error);
        assert_eq!(reported, raw);
    }

    #[tokio::test]
    async fn retry_feedback_tells_json_errors_from_schema_errors() {
        let request = request().with_option(ChatExtractionMode::Native);

        let trailing_comma = MockProvider::build(|b| {
            b.text(r#"{"name":"Alice","age":31,}"#)
                .text(r#"{"name":"Alice","age":31}"#)
        });
        ChatExtractor::new()
            .retries(1)
            .extract::<_, Person>(&trailing_comma, &request)
            .await
            .unwrap();
        let feedback = retry_feedback(&trailing_comma);
        assert!(feedback.contains("it was not valid JSON: "), "{feedback}");
        assert!(!feedback.contains("schema:"), "{feedback}");

        let wrong_type = MockProvider::build(|b| {
            b.text(r#"{"name":"Alice","age":"31"}"#)
                .text(r#"{"name":"Alice","age":31}"#)
        });
        ChatExtractor::new()
            .retries(1)
            .extract::<_, Person>(&wrong_type, &request)
            .await
            .unwrap();
        let feedback = retry_feedback(&wrong_type);
        assert!(
            feedback.contains("it did not match the schema at `age`: invalid type"),
            "{feedback}"
        );

        // Cut-off JSON (an early end of input) is a JSON error too.
        let cut_off = MockProvider::build(|b| {
            b.text(r#"{"name":"Alice""#)
                .text(r#"{"name":"Alice","age":31}"#)
        });
        ChatExtractor::new()
            .retries(1)
            .extract::<_, Person>(&cut_off, &request)
            .await
            .unwrap();
        assert!(retry_feedback(&cut_off).contains("it was not valid JSON: "));
    }

    #[tokio::test]
    async fn forced_tool_retry_feedback_tells_json_errors_from_schema_errors() {
        let provider = MockProvider::build(|b| {
            b.response(ChatResponse::new(vec![crate::ContentBlock::ToolCall(
                crate::ToolCall::new("c1", EXTRACTION_TOOL, r#"{"name":"Alice","#),
            )]))
            .tool_call("c2", EXTRACTION_TOOL, json!({"name": "Alice", "age": 31}))
        });
        let request = request().with_option(ChatExtractionMode::ForcedTool);
        ChatExtractor::new()
            .retries(1)
            .extract::<_, Person>(&provider, &request)
            .await
            .unwrap();
        let retry = provider.last_request().unwrap();
        let result = retry.messages[2].as_tool().unwrap();
        let feedback = format!("{:?}", result.content);
        assert!(feedback.contains("it was not valid JSON: "), "{feedback}");
    }

    /// An order whose line items were extracted with one wrong field.
    #[derive(Debug, PartialEq, Deserialize, JsonSchema)]
    struct Order {
        items: Vec<LineItem>,
    }

    #[derive(Debug, PartialEq, Deserialize, JsonSchema)]
    struct LineItem {
        name: String,
        price: u32,
        #[serde(default)]
        tags: Vec<String>,
    }

    #[tokio::test]
    async fn parse_failures_name_the_field_path() {
        // The stand-in `null`s are removed first, so the path names the
        // wrong price, not the `tags` a strict mode made nullable.
        let raw = r#"{"items":[{"name":"a","price":1,"tags":null},{"name":"b","price":"2","tags":null}]}"#;
        let fixed = r#"{"items":[{"name":"a","price":1},{"name":"b","price":2}]}"#;
        let native = MockProvider::build(|b| b.text(raw).text(fixed));
        let request = request().with_option(ChatExtractionMode::Native);
        ChatExtractor::new()
            .retries(1)
            .extract::<_, Order>(&native, &request)
            .await
            .unwrap();
        let feedback = retry_feedback(&native);
        assert!(
            feedback.contains(
                "it did not match the schema at `items[1].price`: invalid type: string \"2\""
            ),
            "{feedback}"
        );

        // The error holds the path, and its `Display` names it.
        let failure = failure(
            MockProvider::with_text(raw)
                .extract::<Order>(&request)
                .await
                .unwrap_err(),
        );
        let ChatExtractError::Parse { path, message, .. } = failure.error() else {
            panic!("expected a parse failure, got {failure:?}");
        };
        assert_eq!(path.as_deref(), Some("items[1].price"));
        assert_eq!(message, "invalid type: string \"2\", expected u32");
        let shown = failure.error().to_string();
        assert!(
            shown.ends_with(
                "using mode Native: at `items[1].price`: invalid type: string \"2\", expected u32"
            ),
            "{shown}"
        );

        // A missing field names the object that lacks it.
        let missing = parse_failure_of::<Order>(r#"{"items":[{"name":"a"}]}"#).await;
        assert_eq!(missing.0.as_deref(), Some("items[0]"));
        assert!(missing.1.contains("missing field `price`"), "{}", missing.1);

        // No path for invalid JSON or a failure at the root.
        assert_eq!(parse_failure_of::<Order>(r#"{"items":["#).await.0, None);
        assert_eq!(parse_failure_of::<Order>(r#""text""#).await.0, None);
    }

    #[tokio::test]
    async fn forced_tool_feedback_names_the_field_path() {
        let provider = MockProvider::build(|b| {
            b.tool_call("c1", EXTRACTION_TOOL, json!({"name": "Alice", "age": "31"}))
                .tool_call("c2", EXTRACTION_TOOL, json!({"name": "Alice", "age": 31}))
        });
        let request = request().with_option(ChatExtractionMode::ForcedTool);
        ChatExtractor::new()
            .retries(1)
            .extract::<_, Person>(&provider, &request)
            .await
            .unwrap();
        let retry = provider.last_request().unwrap();
        let result = retry.messages[2].as_tool().unwrap();
        let feedback = format!("{:?}", result.content);
        assert!(
            feedback.contains("it did not match the schema at `age`: invalid type"),
            "{feedback}"
        );
    }

    #[tokio::test]
    async fn repaired_output_reports_the_candidate_path() {
        let raw = "Here you go:\n```json\n{\"items\":[{\"name\":\"a\",\"price\":-1}]}\n```";
        let (path, message) =
            parse_failure_with::<Order>(raw, ChatExtractor::new().repair(true)).await;
        assert_eq!(path.as_deref(), Some("items[0].price"));
        assert!(message.contains("invalid value: integer `-1`"), "{message}");
    }

    /// The path and message of the parse failure for a native reply `raw`.
    async fn parse_failure_of<T>(raw: &str) -> (Option<String>, String)
    where
        T: DeserializeOwned + JsonSchema + fmt::Debug + Send + 'static,
    {
        parse_failure_with::<T>(raw, ChatExtractor::new()).await
    }

    async fn parse_failure_with<T>(raw: &str, extractor: ChatExtractor) -> (Option<String>, String)
    where
        T: DeserializeOwned + JsonSchema + fmt::Debug + Send + 'static,
    {
        let provider = MockProvider::with_text(raw);
        let request = request().with_option(ChatExtractionMode::Native);
        let failure = failure(
            extractor
                .extract::<_, T>(&provider, &request)
                .await
                .unwrap_err(),
        );
        let ChatExtractError::Parse { path, message, .. } = failure.error() else {
            panic!("expected a parse failure, got {failure:?}");
        };
        (path.clone(), message.clone())
    }

    #[tokio::test]
    async fn streamed_native_extraction() {
        let provider = MockStreamingProvider::with_text(r#"{"name":"Alice","age":31}"#);
        let request = request().with_option(ChatExtractionMode::Native);
        let mut stream = provider.extract_stream::<Person>(&request).await.unwrap();
        let mut events = 0;
        while let Some(event) = stream.next().await {
            event.unwrap();
            events += 1;
        }
        assert!(events > 0);
        assert_eq!(stream.partial_output(), r#"{"name":"Alice","age":31}"#);
        let extracted = stream.finish().unwrap();
        assert_eq!(extracted.value, alice());
        assert_eq!(extracted.metadata.mode, ChatExtractionMode::Native);
        assert!(provider.requests()[0].response_format.is_some());
    }

    /// A strict mode made the optional properties nullable; the model sent
    /// `null` for each.
    #[derive(Debug, PartialEq, Deserialize, JsonSchema)]
    struct Profile {
        name: String,
        #[serde(default)]
        tags: Vec<String>,
        nickname: Option<String>,
    }

    const STRICT_NULLS: &str = r#"{"name":"Alice","tags":null,"nickname":null}"#;

    fn alice_profile() -> Profile {
        Profile {
            name: "Alice".into(),
            tags: Vec::new(),
            nickname: None,
        }
    }

    #[tokio::test]
    async fn strict_nulls_are_normalized_in_both_modes() {
        let native = MockProvider::with_text(STRICT_NULLS);
        let native_request = request().with_option(ChatExtractionMode::Native);
        let extracted = native.extract::<Profile>(&native_request).await.unwrap();
        assert_eq!(extracted.value, alice_profile());
        assert!(!extracted.metadata.repaired);

        let forced = MockProvider::with_tool_call(
            "c",
            EXTRACTION_TOOL,
            serde_json::from_str::<serde_json::Value>(STRICT_NULLS).unwrap(),
        );
        let forced_request = request().with_option(ChatExtractionMode::ForcedTool);
        let extracted = forced.extract::<Profile>(&forced_request).await.unwrap();
        assert_eq!(extracted.value, alice_profile());

        // Repair and normalization combine.
        let fenced = MockProvider::with_text(format!("```json\n{STRICT_NULLS}\n```"));
        let extracted = ChatExtractor::new()
            .repair(true)
            .extract::<_, Profile>(&fenced, &native_request)
            .await
            .unwrap();
        assert_eq!(extracted.value, alice_profile());
        assert!(extracted.metadata.repaired);
    }

    #[tokio::test]
    async fn strict_nulls_are_normalized_in_streamed_extraction() {
        let native = MockStreamingProvider::with_text(STRICT_NULLS);
        let native_request = request().with_option(ChatExtractionMode::Native);
        let mut stream = native
            .extract_stream::<Profile>(&native_request)
            .await
            .unwrap();
        while stream.next().await.is_some() {}
        assert_eq!(stream.finish().unwrap().value, alice_profile());

        let forced = MockStreamingProvider::with_stream(MockStreamEvent::tool_call_response(
            "c",
            EXTRACTION_TOOL,
            serde_json::from_str::<serde_json::Value>(STRICT_NULLS).unwrap(),
        ));
        let forced_request = request().with_option(ChatExtractionMode::ForcedTool);
        let mut stream = forced
            .extract_stream::<Profile>(&forced_request)
            .await
            .unwrap();
        while stream.next().await.is_some() {}
        assert_eq!(stream.finish().unwrap().value, alice_profile());
    }

    #[tokio::test]
    async fn parse_failures_keep_the_raw_output_the_model_sent() {
        // `name` is required: its null is not a strict-mode stand-in.
        let raw = r#"{"name":null,"tags":null}"#;
        let provider = MockProvider::with_text(raw);
        let request = request().with_option(ChatExtractionMode::Native);
        let err = provider.extract::<Profile>(&request).await.unwrap_err();
        let Error::Extract { failure, .. } = err else {
            panic!("expected extract error");
        };
        assert_eq!(failure.error().raw_output(), Some(raw));
        // serde sees the null, not a missing field.
        let ChatExtractError::Parse { message, .. } = failure.error() else {
            panic!("expected a parse error");
        };
        assert!(message.starts_with("invalid type: null"), "{message}");
    }

    #[tokio::test]
    async fn streamed_forced_tool_extraction_tracks_the_tool_arguments() {
        let provider = MockStreamingProvider::with_stream(MockStreamEvent::tool_call_response(
            "c",
            EXTRACTION_TOOL,
            json!({"name": "Alice", "age": 31}),
        ));
        let request = request().with_option(ChatExtractionMode::ForcedTool);
        let mut stream = provider.extract_stream::<Person>(&request).await.unwrap();
        while stream.next().await.is_some() {}
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(stream.partial_output()).unwrap(),
            json!({"name": "Alice", "age": 31})
        );
        assert_eq!(stream.finish().unwrap().value, alice());
    }

    #[tokio::test]
    async fn streamed_extraction_reports_stream_failures() {
        let provider = MockStreamingProvider::with_stream([
            MockStreamEvent::from(StreamEvent::ResponseStart {
                id: None,
                model: None,
            }),
            MockStreamEvent::from(StreamEvent::TextDelta {
                index: 0,
                text: r#"{"name":"#.into(),
            }),
            MockStreamEvent::from(Error::overloaded("busy")),
        ]);
        let request = request().with_option(ChatExtractionMode::Native);
        let mut stream = provider.extract_stream::<Person>(&request).await.unwrap();
        while stream.next().await.is_some() {}
        assert_eq!(stream.partial_output(), r#"{"name":"#);
        assert!(matches!(stream.finish(), Err(Error::Overloaded { .. })));
    }

    fn failure(err: Error) -> ChatExtractFailure {
        match err {
            Error::Extract { failure, .. } => *failure,
            other => panic!("expected an extraction failure, got {other:?}"),
        }
    }

    fn text_with_usage(text: &str, input: u64, output: u64) -> ChatResponse {
        ChatResponseBuilder::new()
            .text(text)
            .usage(input, output)
            .build()
    }

    #[tokio::test]
    async fn metadata_sums_usage_over_every_pass() {
        let provider = MockProvider::build(|b| {
            b.response(text_with_usage(r#"{"name":"Alice"}"#, 10, 4))
                .response(text_with_usage(r#"{"name":"Alice","age":31}"#, 20, 6))
        });
        let request = request().with_option(ChatExtractionMode::Native);
        let extracted = ChatExtractor::new()
            .retries(1)
            .extract::<_, Person>(&provider, &request)
            .await
            .unwrap();
        assert_eq!(extracted.metadata.passes, 2);
        let usage = extracted.metadata.usage.unwrap();
        assert_eq!(
            (usage.input_tokens, usage.output_tokens),
            (Some(30), Some(10))
        );
        // The response keeps its own usage.
        assert_eq!(extracted.response.usage.unwrap().input_tokens, Some(20));
    }

    #[tokio::test]
    async fn failures_carry_the_last_response_and_the_pass_record() {
        let provider = MockProvider::build(|b| {
            b.response(text_with_usage("nope", 10, 2))
                .response(text_with_usage("still nope", 11, 3))
        });
        let request = request().with_option(ChatExtractionMode::Native);
        let err = ChatExtractor::new()
            .retries(1)
            .extract::<_, Person>(&provider, &request)
            .await
            .unwrap_err();
        assert!(!err.is_retryable());
        let failure = failure(err);
        assert!(matches!(failure.error(), ChatExtractError::Parse { .. }));
        assert_eq!(
            failure.response().and_then(ChatResponse::text).as_deref(),
            Some("still nope")
        );
        assert_eq!(failure.metadata().mode, ChatExtractionMode::Native);
        assert_eq!(failure.metadata().passes, 2);
        assert_eq!(
            failure.metadata().usage.as_ref().unwrap().input_tokens,
            Some(21)
        );
        assert_eq!(
            failure.into_response().unwrap().text().as_deref(),
            Some("still nope")
        );
    }

    #[tokio::test]
    async fn truncated_output_fails_before_parsing_and_is_not_retried() {
        let provider = MockProvider::build(|b| {
            b.response(
                ChatResponseBuilder::new()
                    .text(r#"{"name":"Alice","age":31}"#)
                    .finish_reason(FinishReason::Length)
                    .usage(10, 64)
                    .build(),
            )
            .text(r#"{"name":"Alice","age":31}"#)
        });
        let request = request().with_option(ChatExtractionMode::Native);
        let err = ChatExtractor::new()
            .repair(true)
            .retries(2)
            .extract::<_, Person>(&provider, &request)
            .await
            .unwrap_err();
        assert_eq!(provider.call_count(), 1);
        let failure = failure(err);
        assert!(
            matches!(
                failure.error(),
                ChatExtractError::Truncated {
                    mode: ChatExtractionMode::Native,
                    ..
                }
            ),
            "{failure:?}"
        );
        assert!(failure.to_string().contains("output token limit"));
        assert_eq!(failure.metadata().passes, 1);
        assert_eq!(
            failure.metadata().usage.as_ref().unwrap().output_tokens,
            Some(64)
        );
        assert_eq!(
            failure.response().unwrap().finish_reason,
            Some(FinishReason::Length)
        );
    }

    #[tokio::test]
    async fn a_full_context_window_is_truncation_with_its_own_remedy() {
        let provider = MockProvider::build(|b| {
            b.response(
                ChatResponseBuilder::new()
                    .text(r#"{"name":"Ali"#)
                    .finish_reason(FinishReason::ContextWindow)
                    .usage(190_000, 64)
                    .build(),
            )
            .text(r#"{"name":"Alice","age":31}"#)
        });
        let request = request().with_option(ChatExtractionMode::Native);
        let err = ChatExtractor::new()
            .retries(2)
            .extract::<_, Person>(&provider, &request)
            .await
            .unwrap_err();
        assert!(!err.is_retryable());
        assert_eq!(provider.call_count(), 1);
        let failure = failure(err);
        assert!(
            matches!(
                failure.error(),
                ChatExtractError::Truncated {
                    finish_reason: FinishReason::ContextWindow,
                    ..
                }
            ),
            "{failure:?}"
        );
        let message = failure.to_string();
        assert!(message.contains("context window"), "{message}");
        assert!(message.contains("shorten the input"), "{message}");
        assert!(!message.contains("raise max_output_tokens"), "{message}");
        assert_eq!(failure.metadata().passes, 1);
    }

    #[tokio::test]
    async fn truncation_says_how_much_went_to_reasoning_when_reported() {
        let truncated = |usage: Usage| {
            ChatResponseBuilder::new()
                .finish_reason(FinishReason::Length)
                .usage_value(usage)
                .build()
        };
        let request = request().with_option(ChatExtractionMode::Native);

        let provider = MockProvider::with_response(truncated(
            Usage::new().output_tokens(400).reasoning_tokens(400),
        ));
        let reasoned = failure(provider.extract::<Person>(&request).await.unwrap_err());
        assert!(matches!(
            reasoned.error(),
            ChatExtractError::Truncated {
                output_tokens: Some(400),
                reasoning_tokens: Some(400),
                ..
            }
        ));
        assert!(
            reasoned.error().to_string().ends_with(
                "; 400 of 400 output tokens went to reasoning: raise max_output_tokens or lower the reasoning effort"
            ),
            "{}",
            reasoned.error()
        );

        // Without reported reasoning, nothing is claimed.
        let provider = MockProvider::with_response(truncated(Usage::new().output_tokens(400)));
        let unreported = failure(provider.extract::<Person>(&request).await.unwrap_err());
        assert!(!unreported.error().to_string().contains("reasoning"));
    }

    #[tokio::test]
    async fn truncated_forced_tool_arguments_are_reported() {
        let provider = MockProvider::with_response(
            ChatResponseBuilder::new()
                .tool_call("c", EXTRACTION_TOOL, json!({"name": "Alice"}))
                .finish_reason(FinishReason::Length)
                .build(),
        );
        let request = request().with_option(ChatExtractionMode::ForcedTool);
        let failure = failure(provider.extract::<Person>(&request).await.unwrap_err());
        assert!(matches!(
            failure.error(),
            ChatExtractError::Truncated {
                mode: ChatExtractionMode::ForcedTool,
                ..
            }
        ));
    }

    fn declined(reason: FinishReason) -> ChatResponseBuilder {
        ChatResponseBuilder::new()
            .text("I can't help with that.")
            .finish_reason(reason)
            .usage(10, 6)
    }

    #[tokio::test]
    async fn refusals_fail_before_parsing_and_are_not_retried() {
        for reason in [FinishReason::Refusal, FinishReason::ContentFilter] {
            let provider = MockProvider::build(|b| {
                b.response(declined(reason.clone()).build())
                    .text(r#"{"name":"Alice","age":31}"#)
            });
            let request = request().with_option(ChatExtractionMode::Native);
            let err = ChatExtractor::new()
                .repair(true)
                .retries(2)
                .extract::<_, Person>(&provider, &request)
                .await
                .unwrap_err();
            assert_eq!(provider.call_count(), 1, "{reason}");
            assert!(!err.is_retryable());
            let failure = failure(err);
            let ChatExtractError::Refused {
                mode,
                finish_reason,
                text,
                source,
                ..
            } = failure.error()
            else {
                panic!("a refusal expected: {failure:?}");
            };
            assert_eq!(*mode, ChatExtractionMode::Native);
            assert_eq!(finish_reason.as_ref(), Some(&reason));
            // Nothing explains it: the text the model wrote before the
            // finish is partial output, not the refusal.
            assert_eq!((text, source), (&None, &RefusalSource::None));
            assert_eq!(
                failure.to_string(),
                format!(
                    "mock: the model refused to produce the structured output (finish reason `{reason}`)"
                )
            );
            assert_eq!(failure.metadata().passes, 1);
            assert_eq!(
                failure.metadata().usage.as_ref().unwrap().output_tokens,
                Some(6)
            );
        }
    }

    #[tokio::test]
    async fn a_providers_explanation_is_the_refusal_text() {
        let mut provider_data = crate::ProviderData::new();
        provider_data.insert_raw(
            "acme",
            DECLINE_EXPLANATION_KEY,
            json!("The request touches a restricted topic."),
        );
        let response = ChatResponseBuilder::new()
            .finish_reason(FinishReason::Refusal)
            .build()
            .provider_data(provider_data);
        let provider = MockProvider::with_response(response);
        let request = request().with_option(ChatExtractionMode::ForcedTool);
        let failure = failure(provider.extract::<Person>(&request).await.unwrap_err());
        assert!(
            matches!(
                failure.error(),
                ChatExtractError::Refused {
                    mode: ChatExtractionMode::ForcedTool,
                    text: Some(text),
                    source: RefusalSource::Provider,
                    ..
                } if text == "The request touches a restricted topic."
            ),
            "{failure:?}"
        );
        assert_eq!(
            failure.to_string(),
            "mock: the model refused: The request touches a restricted topic."
        );
        // The redacted log line leaves the upstream text out.
        let log = Error::from(failure).as_log().to_string();
        assert!(!log.contains("restricted"), "{log}");
        assert!(log.contains("finish reason `refusal`"), "{log}");
    }

    #[tokio::test]
    async fn a_streamed_refusal_fails_at_finish() {
        let provider =
            MockStreamingProvider::from_responses([declined(FinishReason::ContentFilter).build()]);
        let request = request().with_option(ChatExtractionMode::Native);
        let mut stream = provider.extract_stream::<Person>(&request).await.unwrap();
        while stream.next().await.is_some() {}
        let failure = failure(stream.finish().unwrap_err());
        assert!(
            matches!(
                failure.error(),
                ChatExtractError::Refused {
                    finish_reason: Some(FinishReason::ContentFilter),
                    ..
                }
            ),
            "{failure:?}"
        );
    }

    #[tokio::test]
    async fn odd_finishes_are_named_in_the_parse_message() {
        let provider = MockProvider::with_response(
            ChatResponseBuilder::new()
                .text("{\"name\":")
                .finish_reason(FinishReason::Other("malformed_function_call".into()))
                .build(),
        );
        let request = request().with_option(ChatExtractionMode::Native);
        let odd = failure(provider.extract::<Person>(&request).await.unwrap_err());
        let ChatExtractError::Parse { message, .. } = odd.error() else {
            panic!("a parse failure expected: {odd:?}");
        };
        assert!(
            message.ends_with("(finish reason `malformed_function_call`)"),
            "{message}"
        );

        // A normal finish adds nothing.
        let provider = MockProvider::with_text("{\"name\":");
        let normal = failure(provider.extract::<Person>(&request).await.unwrap_err());
        let ChatExtractError::Parse { message, .. } = normal.error() else {
            panic!("a parse failure expected: {normal:?}");
        };
        assert!(!message.contains("finish reason"), "{message}");
    }

    #[tokio::test]
    async fn blank_output_is_missing_structured_text() {
        for text in ["", "  \n\t "] {
            let provider = MockProvider::with_text(text);
            let request = request().with_option(ChatExtractionMode::Native);
            let failure = failure(provider.extract::<Person>(&request).await.unwrap_err());
            assert!(
                matches!(
                    failure.error(),
                    ChatExtractError::MissingStructuredText { .. }
                ),
                "{text:?}: {failure:?}"
            );
        }
    }

    #[test]
    fn refusal_errors_round_trip() {
        let error = ChatExtractError::Refused {
            mode: ChatExtractionMode::Native,
            provider: "acme".into(),
            finish_reason: Some(FinishReason::ContentFilter),
            text: Some("filtered".into()),
            source: RefusalSource::Provider,
        };
        let json = serde_json::to_value(&error).unwrap();
        assert_eq!(json["type"], "refused");
        assert_eq!(json["source"], "provider");
        let back: ChatExtractError = serde_json::from_value(json).unwrap();
        assert_eq!(back.to_string(), "acme: the model refused: filtered");
    }

    fn rate_limited() -> Error {
        Error::rate_limited("HTTP 429")
            .authored()
            .with_provider("mock")
            .with_status(429)
            .with_retry_after(std::time::Duration::from_secs(7))
    }

    #[tokio::test]
    async fn provider_errors_on_a_retry_carry_the_earlier_passes() {
        let provider = MockProvider::build(|b| {
            b.response(text_with_usage(r#"{"name":"Alice"}"#, 500, 50))
                .error(rate_limited())
        });
        let request = request().with_option(ChatExtractionMode::Native);
        let err = ChatExtractor::new()
            .retries(1)
            .extract::<_, Person>(&provider, &request)
            .await
            .unwrap_err();

        // The provider error is returned unchanged, not reclassified.
        assert!(matches!(err, Error::RateLimited { .. }), "{err:?}");
        assert_eq!(err.kind_name(), "rate_limited");
        assert!(err.is_retryable());
        assert_eq!(err.retry_after(), Some(std::time::Duration::from_secs(7)));

        let metadata = ChatExtractionMetadata::from_error(&err).unwrap();
        assert_eq!(metadata.mode, ChatExtractionMode::Native);
        assert_eq!(metadata.passes, 1);
        let usage = metadata.usage.unwrap();
        assert_eq!(
            (usage.input_tokens, usage.output_tokens),
            (Some(500), Some(50))
        );
    }

    #[tokio::test]
    async fn provider_errors_on_the_first_pass_carry_an_empty_record() {
        let provider = MockProvider::build(|b| b.error(rate_limited()));
        let request = request().with_option(ChatExtractionMode::ForcedTool);
        let err = provider.extract::<Person>(&request).await.unwrap_err();
        assert!(matches!(err, Error::RateLimited { .. }), "{err:?}");
        assert_eq!(
            ChatExtractionMetadata::from_error(&err),
            Some(ChatExtractionMetadata::new(ChatExtractionMode::ForcedTool))
        );
    }

    #[tokio::test]
    async fn stream_setup_errors_carry_an_empty_record() {
        let provider = MockStreamingProvider::build(|b| b.setup_error(rate_limited()));
        let request = request().with_option(ChatExtractionMode::Native);
        let Err(err) = provider.extract_stream::<Person>(&request).await else {
            panic!("expected a setup error");
        };
        assert!(matches!(err, Error::RateLimited { .. }), "{err:?}");
        assert_eq!(err.retry_after(), Some(std::time::Duration::from_secs(7)));
        assert_eq!(
            ChatExtractionMetadata::from_error(&err),
            Some(ChatExtractionMetadata::new(ChatExtractionMode::Native))
        );
    }

    #[tokio::test]
    async fn stream_errors_returned_by_finish_carry_the_record() {
        let provider = MockStreamingProvider::with_stream([
            MockStreamEvent::from(StreamEvent::ResponseStart {
                id: None,
                model: None,
            }),
            MockStreamEvent::from(Error::overloaded("busy")),
        ]);
        let request = request().with_option(ChatExtractionMode::Native);
        let mut stream = provider.extract_stream::<Person>(&request).await.unwrap();
        while stream.next().await.is_some() {}
        let err = stream.finish().unwrap_err();
        assert!(matches!(err, Error::Overloaded { .. }), "{err:?}");
        assert!(err.is_retryable());
        assert_eq!(
            ChatExtractionMetadata::from_error(&err),
            Some(ChatExtractionMetadata::new(ChatExtractionMode::Native))
        );
    }

    /// A stream that reported a degradation and its usage, then failed
    /// before finishing.
    fn failing_mid_stream() -> Vec<MockStreamEvent> {
        vec![
            MockStreamEvent::from(StreamEvent::response_start(None, None)),
            MockStreamEvent::from(StreamEvent::warning(strict_dropped(
                RequestField::ResponseFormat,
            ))),
            MockStreamEvent::from(StreamEvent::text_delta(0, r#"{"name":"#)),
            MockStreamEvent::from(
                crate::chat::stream::MetadataBuilder::new()
                    .with_usage(Usage::new().input_tokens(900).output_tokens(4))
                    .build(),
            ),
            MockStreamEvent::from(Error::overloaded("busy")),
        ]
    }

    #[tokio::test]
    async fn a_stream_failing_mid_way_keeps_its_usage_and_degradations() {
        let provider = MockStreamingProvider::with_stream(failing_mid_stream());
        let request = request().with_option(ChatExtractionMode::Native);
        let mut stream = provider.extract_stream::<Person>(&request).await.unwrap();
        while stream.next().await.is_some() {}
        let err = stream.finish().unwrap_err();
        assert!(matches!(err, Error::Overloaded { .. }), "{err:?}");
        let record = ChatExtractionMetadata::from_error(&err).unwrap();
        // Billed, but not a pass: the stream returned no response.
        assert_eq!(record.passes, 0);
        assert_eq!(
            record.usage.as_ref().and_then(|u| u.input_tokens),
            Some(900)
        );
        assert_eq!(record.usage.as_ref().and_then(|u| u.output_tokens), Some(4));
        assert_eq!(
            record.degradations,
            [strict_dropped(RequestField::ResponseFormat)]
        );
        assert_eq!(record.enforcement, SchemaEnforcement::NotEnforced);
    }

    #[tokio::test]
    async fn a_stream_ending_early_keeps_its_usage() {
        // No error item: the stream just stops before `ResponseStop`.
        let mut events = failing_mid_stream();
        events.pop();
        let provider = MockStreamingProvider::with_stream(events);
        let request = request().with_option(ChatExtractionMode::Native);
        let mut stream = provider.extract_stream::<Person>(&request).await.unwrap();
        while stream.next().await.is_some() {}
        let err = stream.finish().unwrap_err();
        assert!(matches!(err, Error::Stream { .. }), "{err:?}");
        let record = ChatExtractionMetadata::from_error(&err).unwrap();
        assert_eq!(record.passes, 0);
        assert_eq!(record.usage.and_then(|u| u.input_tokens), Some(900));
    }

    #[tokio::test]
    async fn a_failed_stream_after_a_fallback_keeps_both_records() {
        let mut events = vec![
            MockStreamEvent::from(StreamEvent::response_start(None, None)),
            MockStreamEvent::from(StreamEvent::tool_call_start(0, "c", EXTRACTION_TOOL)),
            MockStreamEvent::from(StreamEvent::tool_call_delta(0, r#"{"name":"#)),
        ];
        events.extend(failing_mid_stream().into_iter().skip(3));
        let provider = MockStreamingProvider::build(|b| {
            b.setup_error(refused(RequestField::ResponseFormat))
                .stream(events)
        })
        .with_chat_capability(
            ChatCapability::StructuredOutput,
            CapabilitySupport::Supported,
        );
        let mut stream = provider.extract_stream::<Person>(&request()).await.unwrap();
        while stream.next().await.is_some() {}
        let err = stream.finish().unwrap_err();
        let record = ChatExtractionMetadata::from_error(&err).unwrap();
        assert_eq!(record.mode, ChatExtractionMode::ForcedTool);
        assert_eq!(record.auto_fallback, Some(refusal_record()));
        assert_eq!(record.passes, 0);
        assert_eq!(record.usage.and_then(|u| u.input_tokens), Some(900));
    }

    #[tokio::test]
    async fn a_failed_failover_carries_the_record_on_its_own_context() {
        let provider = MockProvider::build(|b| {
            b.response(text_with_usage("nope", 10, 2))
                .error(rate_limited().with_fallback_from(Error::timeout("primary")))
        });
        let request = request().with_option(ChatExtractionMode::Native);
        let err = ChatExtractor::new()
            .retries(1)
            .extract::<_, Person>(&provider, &request)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::RateLimited { .. }), "{err:?}");
        assert_eq!(ChatExtractionMetadata::from_error(&err).unwrap().passes, 1);
        let primary = err.fallback_from().unwrap();
        assert!(matches!(primary, Error::Timeout { .. }), "{primary:?}");
        assert!(ChatExtractionMetadata::from_error(primary).is_none());
    }

    #[tokio::test]
    async fn errors_without_a_record_report_none() {
        assert_eq!(ChatExtractionMetadata::from_error(&rate_limited()), None);
        // A variant without a context cannot carry the record.
        let provider = MockProvider::build(|b| {
            b.error(Error::serialization(
                serde_json::from_str::<serde_json::Value>("{").unwrap_err(),
            ))
        });
        let request = request().with_option(ChatExtractionMode::Native);
        let err = provider.extract::<Person>(&request).await.unwrap_err();
        assert!(matches!(err, Error::Serialization { .. }));
        assert_eq!(ChatExtractionMetadata::from_error(&err), None);
    }

    #[tokio::test]
    async fn error_log_keeps_the_extraction_record() {
        let provider = MockProvider::build(|b| {
            b.response(text_with_usage(r#"{"name":"Alice"}"#, 500, 50))
                .error(rate_limited().with_body("secret body"))
        });
        let request = request().with_option(ChatExtractionMode::Native);
        let mut err = ChatExtractor::new()
            .retries(1)
            .extract::<_, Person>(&provider, &request)
            .await
            .unwrap_err();
        err.context_mut()
            .unwrap()
            .metadata
            .insert("vendor.detail".into(), json!("may quote input"));

        let log = serde_json::to_value(err.as_log()).unwrap();
        assert_eq!(log["allama.chat.extraction"]["passes"], 1);
        assert_eq!(log["allama.chat.extraction"]["mode"], "native");
        assert_eq!(log["allama.chat.extraction"]["usage"]["input_tokens"], 500);
        assert!(log.get("vendor.detail").is_none());
        assert!(!log.to_string().contains("may quote input"));
        assert_eq!(
            err.as_log().to_string(),
            "kind=rate_limited provider=mock status=429 body_len=11 \
             allama.chat.extraction.mode=native allama.chat.extraction.passes=1 \
             allama.chat.extraction.input_tokens=500 allama.chat.extraction.output_tokens=50 \
             message=\"HTTP 429\""
        );
    }

    #[tokio::test]
    async fn missing_text_after_tool_calls_keeps_the_response() {
        let provider = MockProvider::with_tool_call("c1", "lookup", json!({"q": "Alice"}))
            .with_chat_capability(
                ChatCapability::StructuredOutput,
                CapabilitySupport::Supported,
            );
        let failure = failure(provider.extract::<Person>(&request()).await.unwrap_err());
        assert!(matches!(
            failure.error(),
            ChatExtractError::MissingStructuredText { .. }
        ));
        let calls: Vec<_> = failure.response().unwrap().tool_calls().collect();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "lookup");
    }

    #[tokio::test]
    async fn extract_keeps_sampling_settings() {
        let provider = MockProvider::with_text(r#"{"name":"Alice","age":31}"#);
        let request = request()
            .temperature(0.0)
            .seed(7)
            .with_option(ChatExtractionMode::Native);
        ChatExtractor::new()
            .extract::<_, Person>(&provider, &request)
            .await
            .unwrap();
        let sent = provider.last_request().unwrap();
        assert_eq!((sent.temperature, sent.seed), (Some(0.0), Some(7)));
    }

    #[test]
    fn parse_uses_an_existing_response() {
        let response = ChatResponseBuilder::new()
            .text("Here: {\"name\":\"Alice\",\"age\":31}")
            .usage(5, 5)
            .build();
        let extracted = ChatExtractor::new()
            .repair(true)
            .parse::<Person>(response.clone(), ChatExtractionMode::Native)
            .unwrap();
        assert_eq!(extracted.value, alice());
        assert_eq!(extracted.response, response);
        assert_eq!(
            extracted.metadata,
            ChatExtractionMetadata {
                mode: ChatExtractionMode::Native,
                passes: 0,
                repaired: true,
                enforcement: SchemaEnforcement::NotEnforced,
                degradations: Vec::new(),
                usage: None,
                auto_fallback: None,
                undeclared_key_rejections: 0,
            }
        );

        let strict = failure(
            ChatExtractor::new()
                .parse::<Person>(response.clone(), ChatExtractionMode::Native)
                .unwrap_err(),
        );
        assert!(matches!(strict.error(), ChatExtractError::Parse { .. }));
        assert_eq!(strict.response(), Some(&response));

        let tool = ChatResponseBuilder::new()
            .tool_call("c", EXTRACTION_TOOL, json!({"name": "Alice", "age": 31}))
            .build();
        let extracted = ChatExtractor::new()
            .parse::<Person>(tool, ChatExtractionMode::ForcedTool)
            .unwrap();
        assert_eq!(extracted.value, alice());
    }

    #[test]
    fn parse_rejects_auto() {
        let mut response = ChatResponse::new(vec![crate::ContentBlock::text("{}")]);
        response.provider = Some("demo".into());
        let failure = failure(
            ChatExtractor::new()
                .parse::<Person>(response, ChatExtractionMode::Auto)
                .unwrap_err(),
        );
        assert!(matches!(
            failure.error(),
            ChatExtractError::Unsupported { provider, .. } if provider == "demo"
        ));
        assert!(failure.response().is_some());
    }

    #[tokio::test]
    async fn streamed_failures_carry_the_collected_response() {
        let provider = MockStreamingProvider::from_responses([text_with_usage("nope", 3, 1)]);
        let request = request().with_option(ChatExtractionMode::Native);
        let mut stream = provider.extract_stream::<Person>(&request).await.unwrap();
        while stream.next().await.is_some() {}
        let failure = failure(stream.finish().unwrap_err());
        assert!(matches!(failure.error(), ChatExtractError::Parse { .. }));
        assert_eq!(failure.metadata().passes, 1);
        assert_eq!(
            failure.metadata().usage.as_ref().unwrap().input_tokens,
            Some(3)
        );
        assert_eq!(
            failure.response().and_then(ChatResponse::text).as_deref(),
            Some("nope")
        );
    }

    #[tokio::test]
    async fn error_log_omits_the_failed_response() {
        let provider = MockProvider::with_text("secret model output");
        let request = request().with_option(ChatExtractionMode::Native);
        let err = provider.extract::<Person>(&request).await.unwrap_err();

        let log = serde_json::to_value(err.as_log()).unwrap();
        assert_eq!(log["type"], "extract");
        assert!(!log.to_string().contains("secret model output"), "{log}");
        assert!(!err.as_log().to_string().contains("secret model output"));

        // The full serde form keeps it, and round-trips.
        let full = serde_json::to_value(&err).unwrap();
        assert!(full.to_string().contains("secret model output"));
        let back = failure(serde_json::from_value(full).unwrap());
        assert_eq!(
            back.response().and_then(ChatResponse::text).as_deref(),
            Some("secret model output")
        );
        assert_eq!(back.metadata().passes, 1);
    }

    #[tokio::test]
    async fn the_synthetic_tool_is_strict() {
        let provider =
            MockProvider::with_tool_call("c", EXTRACTION_TOOL, json!({"name": "Alice", "age": 31}));
        let request = request().with_option(ChatExtractionMode::ForcedTool);
        let extracted = provider.extract::<Person>(&request).await.unwrap();
        assert_eq!(extracted.metadata.enforcement, SchemaEnforcement::Exact);
        let sent = provider.last_request().unwrap();
        let tools = sent.tools;
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, EXTRACTION_TOOL);
        assert_eq!(tools[0].strict, Some(true));
        assert_eq!(sent.parallel_tool_calls, None);
    }

    /// Extraction never sets `parallel_tool_calls`, whatever the model id:
    /// the caller's value reaches the adapter, which applies its dialect's
    /// rule.
    #[tokio::test]
    async fn the_callers_parallel_tool_calls_is_kept() {
        for model in ["gpt-4.1", "ft:gpt-4o-mini-2024-07-18:acme::abc123"] {
            for (setting, mode) in [
                (None, ChatExtractionMode::ForcedTool),
                (Some(true), ChatExtractionMode::ForcedTool),
                (Some(false), ChatExtractionMode::ForcedTool),
                (Some(true), ChatExtractionMode::Native),
            ] {
                let provider = if mode == ChatExtractionMode::Native {
                    MockProvider::with_text(r#"{"name":"Alice","age":31}"#)
                } else {
                    MockProvider::with_tool_call(
                        "c",
                        EXTRACTION_TOOL,
                        json!({"name": "Alice", "age": 31}),
                    )
                };
                let mut request = ChatRequest::new(model)
                    .user("Alice is 31")
                    .with_option(mode);
                request.parallel_tool_calls = setting;
                provider.extract::<Person>(&request).await.unwrap();
                assert_eq!(
                    provider.last_request().unwrap().parallel_tool_calls,
                    setting,
                    "{model} {mode:?}"
                );
            }
        }
    }

    fn strict_dropped(field: RequestField) -> Warning {
        Warning::unsupported(field, "strict is not supported; sent without strict")
            .with_provider("mock")
    }

    #[tokio::test]
    async fn unsupported_degradations_are_reported_as_not_enforced() {
        let provider = MockProvider::with_response(
            ChatResponseBuilder::new()
                .tool_call("c", EXTRACTION_TOOL, json!({"name": "Alice", "age": 31}))
                .warning(strict_dropped(RequestField::Tools))
                .warning(Warning::unsupported(RequestField::Seed, "seed dropped"))
                .build(),
        );
        let request = request().with_option(ChatExtractionMode::ForcedTool);
        let extracted = provider.extract::<Person>(&request).await.unwrap();
        assert_eq!(extracted.value, alice());
        assert_eq!(
            extracted.metadata.enforcement,
            SchemaEnforcement::NotEnforced
        );
        // Only tool warnings count in `ForcedTool` mode.
        assert_eq!(
            extracted.metadata.degradations,
            [strict_dropped(RequestField::Tools)]
        );
        assert_eq!(extracted.response.warnings.len(), 2);
    }

    fn approximated(field: RequestField) -> Warning {
        Warning::new(
            WarningKind::Approximated,
            Subject::Field { field },
            "`pattern` was removed",
        )
        .with_provider("mock")
    }

    #[tokio::test]
    async fn approximated_degradations_are_reported_as_approximate() {
        let provider = MockProvider::with_response(
            ChatResponseBuilder::new()
                .text(r#"{"name":"Alice","age":31}"#)
                .warning(approximated(RequestField::ResponseFormat))
                .build(),
        );
        let request = request().with_option(ChatExtractionMode::Native);
        let extracted = provider.extract::<Person>(&request).await.unwrap();
        assert_eq!(
            extracted.metadata.enforcement,
            SchemaEnforcement::Approximate
        );
        assert_eq!(
            extracted.metadata.degradations,
            [approximated(RequestField::ResponseFormat)]
        );
    }

    #[tokio::test]
    async fn only_the_field_the_mode_relies_on_counts() {
        // Native mode: a tools warning leaves the response format enforced.
        let provider = MockProvider::with_response(
            ChatResponseBuilder::new()
                .text(r#"{"name":"Alice","age":31}"#)
                .warning(strict_dropped(RequestField::Tools))
                .build(),
        );
        let native = request().with_option(ChatExtractionMode::Native);
        let extracted = provider.extract::<Person>(&native).await.unwrap();
        assert_eq!(extracted.metadata.enforcement, SchemaEnforcement::Exact);
        assert!(extracted.metadata.degradations.is_empty());
        assert_eq!(extracted.response.warnings.len(), 1);

        // `ForcedTool` mode: a response-format warning does not count.
        let provider = MockProvider::with_response(
            ChatResponseBuilder::new()
                .tool_call("c", EXTRACTION_TOOL, json!({"name": "Alice", "age": 31}))
                .warning(strict_dropped(RequestField::ResponseFormat))
                .build(),
        );
        let request = request().with_option(ChatExtractionMode::ForcedTool);
        let extracted = provider.extract::<Person>(&request).await.unwrap();
        assert_eq!(extracted.metadata.enforcement, SchemaEnforcement::Exact);
        assert!(extracted.metadata.degradations.is_empty());
    }

    #[tokio::test]
    async fn the_least_enforced_pass_wins() {
        let provider = MockProvider::build(|b| {
            b.response(
                ChatResponseBuilder::new()
                    .text("not json")
                    .warning(approximated(RequestField::ResponseFormat))
                    .build(),
            )
            .response(
                ChatResponseBuilder::new()
                    .text(r#"{"name":"Alice","age":31}"#)
                    .warning(strict_dropped(RequestField::ResponseFormat))
                    .build(),
            )
        });
        let request = request().with_option(ChatExtractionMode::Native);
        let extracted = ChatExtractor::new()
            .retries(1)
            .extract::<_, Person>(&provider, &request)
            .await
            .unwrap();
        assert_eq!(
            extracted.metadata.enforcement,
            SchemaEnforcement::NotEnforced
        );
        assert_eq!(extracted.metadata.degradations.len(), 2);
    }

    #[test]
    fn enforcement_levels_serialize_in_snake_case() {
        for (level, name) in [
            (SchemaEnforcement::Exact, "exact"),
            (SchemaEnforcement::Approximate, "approximate"),
            (SchemaEnforcement::NotEnforced, "not_enforced"),
        ] {
            assert_eq!(serde_json::to_value(level).unwrap(), name);
            assert_eq!(
                serde_json::from_value::<SchemaEnforcement>(json!(name)).unwrap(),
                level
            );
        }
    }

    #[tokio::test]
    async fn degradations_are_collected_once_over_every_pass() {
        let degraded = |text: &str| {
            ChatResponseBuilder::new()
                .text(text)
                .warning(strict_dropped(RequestField::ResponseFormat))
                .build()
        };
        let provider = MockProvider::build(|b| {
            b.response(degraded(r#"{"name":"Alice"}"#))
                .response(degraded(r#"{"name":"Alice","age":31}"#))
        });
        let request = request().with_option(ChatExtractionMode::Native);
        let extracted = ChatExtractor::new()
            .retries(1)
            .extract::<_, Person>(&provider, &request)
            .await
            .unwrap();
        assert_eq!(extracted.metadata.passes, 2);
        assert_eq!(
            extracted.metadata.enforcement,
            SchemaEnforcement::NotEnforced
        );
        assert_eq!(
            extracted.metadata.degradations,
            [strict_dropped(RequestField::ResponseFormat)]
        );
    }

    #[tokio::test]
    async fn extraction_failures_carry_the_degradations() {
        let provider = MockProvider::with_response(
            ChatResponseBuilder::new()
                .text("not json")
                .warning(strict_dropped(RequestField::ResponseFormat))
                .build(),
        );
        let request = request().with_option(ChatExtractionMode::Native);
        let failure = failure(provider.extract::<Person>(&request).await.unwrap_err());
        assert!(matches!(failure.error(), ChatExtractError::Parse { .. }));
        assert_eq!(
            failure.metadata().enforcement,
            SchemaEnforcement::NotEnforced
        );
        assert_eq!(
            failure.metadata().degradations,
            [strict_dropped(RequestField::ResponseFormat)]
        );

        // Nothing was sent, so nothing was enforced.
        let provider = MockProvider::empty()
            .with_chat_capability(
                ChatCapability::StructuredOutput,
                CapabilitySupport::Unsupported,
            )
            .with_chat_capability(ChatCapability::ToolCalls, CapabilitySupport::Unsupported);
        let failure = failure_of(provider.extract::<Person>(&ChatRequest::new("m")).await);
        assert_eq!(failure.metadata().passes, 0);
        assert_eq!(
            failure.metadata().enforcement,
            SchemaEnforcement::NotEnforced
        );
        assert!(failure.metadata().degradations.is_empty());
    }

    fn failure_of<T: fmt::Debug>(result: Result<T>) -> ChatExtractFailure {
        failure(result.unwrap_err())
    }

    #[tokio::test]
    async fn streamed_extraction_reports_the_streamed_degradations() {
        let degraded = |text: &str| {
            ChatResponseBuilder::new()
                .text(text)
                .warning(strict_dropped(RequestField::ResponseFormat))
                .build()
        };
        let request = request().with_option(ChatExtractionMode::Native);

        let provider =
            MockStreamingProvider::from_responses([degraded(r#"{"name":"Alice","age":31}"#)]);
        let mut stream = provider.extract_stream::<Person>(&request).await.unwrap();
        while stream.next().await.is_some() {}
        let extracted = stream.finish().unwrap();
        assert_eq!(
            extracted.metadata.enforcement,
            SchemaEnforcement::NotEnforced
        );
        assert_eq!(
            extracted.metadata.degradations,
            [strict_dropped(RequestField::ResponseFormat)]
        );

        let provider = MockStreamingProvider::from_responses([degraded("nope")]);
        let mut stream = provider.extract_stream::<Person>(&request).await.unwrap();
        while stream.next().await.is_some() {}
        let failure = failure(stream.finish().unwrap_err());
        assert_eq!(
            failure.metadata().enforcement,
            SchemaEnforcement::NotEnforced
        );
        assert_eq!(
            failure.metadata().degradations,
            [strict_dropped(RequestField::ResponseFormat)]
        );

        let provider = MockStreamingProvider::with_text(r#"{"name":"Alice","age":31}"#);
        let mut stream = provider.extract_stream::<Person>(&request).await.unwrap();
        while stream.next().await.is_some() {}
        let extracted = stream.finish().unwrap();
        assert_eq!(extracted.metadata.enforcement, SchemaEnforcement::Exact);
        assert!(extracted.metadata.degradations.is_empty());
    }

    #[test]
    fn parse_normalizes_strict_nulls_and_keeps_the_raw_output() {
        let response = ChatResponseBuilder::new().text(STRICT_NULLS).build();
        let extracted = ChatExtractor::new()
            .parse::<Profile>(response, ChatExtractionMode::Native)
            .unwrap();
        assert_eq!(extracted.value, alice_profile());
        assert_eq!(
            extracted.metadata.enforcement,
            SchemaEnforcement::NotEnforced
        );

        let tool = ChatResponseBuilder::new()
            .tool_call(
                "c",
                EXTRACTION_TOOL,
                serde_json::from_str::<serde_json::Value>(STRICT_NULLS).unwrap(),
            )
            .build();
        let extracted = ChatExtractor::new()
            .parse::<Profile>(tool, ChatExtractionMode::ForcedTool)
            .unwrap();
        assert_eq!(extracted.value, alice_profile());

        // `name` is required: its null is not a stand-in, and the raw output
        // is the text the model sent, not the normalized value.
        let raw = r#"{"name":null,"tags":null}"#;
        let failure = failure_of(ChatExtractor::new().parse::<Profile>(
            ChatResponseBuilder::new().text(raw).build(),
            ChatExtractionMode::Native,
        ));
        assert_eq!(failure.error().raw_output(), Some(raw));
    }

    // --- Undeclared keys ---

    /// A response the adapter reported approximating: its output is checked
    /// for undeclared keys.
    fn approximate_text(text: &str) -> ChatResponse {
        ChatResponseBuilder::new()
            .text(text)
            .warning(approximated(RequestField::ResponseFormat))
            .usage(10, 5)
            .build()
    }

    #[derive(Debug, PartialEq, serde::Deserialize, schemars::JsonSchema)]
    struct Contact {
        email: Option<String>,
        #[serde(default)]
        phone: Option<String>,
    }

    #[derive(Debug, PartialEq, serde::Deserialize, schemars::JsonSchema)]
    struct Customer {
        name: String,
        contact: Option<Contact>,
    }

    #[tokio::test]
    async fn a_misnamed_key_is_rejected_on_a_pass_that_is_not_exact() {
        let provider = MockProvider::with_response(approximate_text(
            r#"{"name": "Ada", "contact": {"e_mail": "ada@example.com"}}"#,
        ));
        let request = request().with_option(ChatExtractionMode::Native);
        let failure = failure(provider.extract::<Customer>(&request).await.unwrap_err());
        let ChatExtractError::Parse {
            message, path, raw, ..
        } = failure.error()
        else {
            panic!("a parse failure expected: {failure:?}");
        };
        assert_eq!(path.as_deref(), Some("contact.e_mail"));
        assert_eq!(
            message,
            "unknown field `e_mail`, expected one of `email`, `phone`"
        );
        assert!(raw.contains("e_mail"));
        assert_eq!(failure.metadata().undeclared_key_rejections, 1);
        assert_eq!(
            failure.metadata().enforcement,
            SchemaEnforcement::Approximate
        );
    }

    #[tokio::test]
    async fn an_undeclared_key_is_retried_and_counted() {
        let provider = MockProvider::build(|b| {
            b.response(approximate_text(r#"{"name": "Ada", "e_mail": "x"}"#))
                .response(approximate_text(r#"{"name": "Ada", "contact": null}"#))
        });
        let request = request().with_option(ChatExtractionMode::Native);
        let extracted = ChatExtractor::new()
            .retries(1)
            .extract::<_, Customer>(&provider, &request)
            .await
            .unwrap();
        assert_eq!(extracted.value.name, "Ada");
        assert_eq!(extracted.metadata.passes, 2);
        assert_eq!(extracted.metadata.undeclared_key_rejections, 1);
        let feedback = retry_feedback(&provider);
        assert!(
            feedback.contains("it did not match the schema at `e_mail`: unknown field `e_mail`"),
            "{feedback}"
        );
    }

    #[tokio::test]
    async fn a_key_nested_at_the_wrong_level_is_rejected() {
        let provider = MockProvider::with_response(approximate_text(
            r#"{"name": "Ada", "email": "ada@example.com"}"#,
        ));
        let request = request().with_option(ChatExtractionMode::Native);
        let failure = failure(provider.extract::<Customer>(&request).await.unwrap_err());
        assert!(
            matches!(failure.error(), ChatExtractError::Parse { path: Some(path), .. } if path == "email"),
            "{failure:?}"
        );
    }

    #[derive(Debug, PartialEq, serde::Deserialize, schemars::JsonSchema)]
    #[serde(untagged)]
    enum Reach {
        Email { email: Option<String> },
        Phone { phone: Option<String> },
    }

    #[derive(Debug, PartialEq, serde::Deserialize, schemars::JsonSchema)]
    struct Lead {
        reach: Reach,
    }

    /// serde alone would read `{"e_mail": ..}` as `Email { email: None }`.
    #[tokio::test]
    async fn untagged_enums_with_optional_branches_are_checked_against_every_branch() {
        let request = request().with_option(ChatExtractionMode::Native);
        let provider =
            MockProvider::with_response(approximate_text(r#"{"reach": {"e_mail": "x"}}"#));
        let failure = failure(provider.extract::<Lead>(&request).await.unwrap_err());
        assert!(
            matches!(failure.error(), ChatExtractError::Parse { path: Some(path), .. } if path == "reach.e_mail"),
            "{failure:?}"
        );
        let provider =
            MockProvider::with_response(approximate_text(r#"{"reach": {"phone": "1"}}"#));
        // A key some branch declares passes. Which branch serde picks is
        // serde's: the first one that deserializes, here `Email`, which
        // ignores `phone` (the documented limit of the union rule).
        let extracted = provider.extract::<Lead>(&request).await.unwrap();
        assert_eq!(extracted.value.reach, Reach::Email { email: None });
    }

    #[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
    struct Tagged {
        labels: std::collections::BTreeMap<String, String>,
        #[serde(flatten)]
        extra: std::collections::BTreeMap<String, serde_json::Value>,
    }

    #[tokio::test]
    async fn maps_and_flattened_maps_take_any_key() {
        let provider = MockProvider::with_response(approximate_text(
            r#"{"labels": {"any": "x"}, "unlisted": 1}"#,
        ));
        let request = request().with_option(ChatExtractionMode::Native);
        let extracted = provider.extract::<Tagged>(&request).await.unwrap();
        assert_eq!(extracted.value.labels["any"], "x");
        assert_eq!(extracted.value.extra["unlisted"], 1);
        assert_eq!(extracted.metadata.undeclared_key_rejections, 0);
    }

    #[tokio::test]
    async fn undeclared_keys_can_be_allowed_and_exact_passes_are_not_checked() {
        let output = r#"{"name": "Ada", "e_mail": "x"}"#;
        let request = request().with_option(ChatExtractionMode::Native);
        let provider = MockProvider::with_response(approximate_text(output));
        let extracted = ChatExtractor::new()
            .allow_undeclared_keys(true)
            .extract::<_, Customer>(&provider, &request)
            .await
            .unwrap();
        assert_eq!(extracted.value.name, "Ada");

        // No degradation: the provider constrained decoding to the schema.
        let provider = MockProvider::with_text(output);
        let extracted = provider.extract::<Customer>(&request).await.unwrap();
        assert_eq!(extracted.metadata.enforcement, SchemaEnforcement::Exact);
    }

    #[test]
    fn parse_rejects_undeclared_keys_unless_allowed() {
        let response = || {
            ChatResponse::new(vec![crate::ContentBlock::text(
                r#"{"name": "Ada", "x": 1}"#,
            )])
        };
        let err = ChatExtractor::new()
            .parse::<Customer>(response(), ChatExtractionMode::Native)
            .unwrap_err();
        let failure = failure(err);
        assert_eq!(failure.metadata().undeclared_key_rejections, 1);
        assert!(
            ChatExtractor::new()
                .allow_undeclared_keys(true)
                .parse::<Customer>(response(), ChatExtractionMode::Native)
                .is_ok()
        );
    }

    #[tokio::test]
    async fn streamed_extraction_rejects_undeclared_keys() {
        let provider = MockStreamingProvider::from_responses([approximate_text(
            r#"{"name": "Ada", "e_mail": "x"}"#,
        )]);
        let request = request().with_option(ChatExtractionMode::Native);
        let mut stream = provider.extract_stream::<Customer>(&request).await.unwrap();
        while stream.next().await.is_some() {}
        let failure = failure(stream.finish().unwrap_err());
        assert_eq!(failure.metadata().undeclared_key_rejections, 1);
    }

    // --- Enforcement in the caller's terms, the floor and the plan ---

    /// A provider whose validation is a dry run that reports `warnings`
    /// (`Encoding` coverage), or a plain one (`Capabilities`).
    struct Validating {
        inner: MockProvider,
        warnings: Vec<Warning>,
        coverage: crate::ValidationCoverage,
    }

    impl crate::ProviderIdentity for Validating {
        fn provider_name(&self) -> &str {
            "mock"
        }

        fn reads_namespace(&self, namespace: &str) -> bool {
            namespace == "mock"
        }
    }

    impl ChatProvider for Validating {
        type Stream = <MockProvider as ChatProvider>::Stream;

        async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse> {
            self.inner.chat(request).await
        }

        async fn chat_stream(&self, request: &ChatRequest) -> Result<Self::Stream> {
            self.inner.chat_stream(request).await
        }

        fn validate_chat_request(&self, _: &ChatRequest) -> Result<crate::ChatValidation> {
            let mut validation = crate::Validation::new(self.coverage);
            for warning in &self.warnings {
                validation = validation.with_warning(warning.clone());
            }
            Ok(validation)
        }

        fn chat_capability(&self, model: &str, capability: ChatCapability) -> CapabilitySupport {
            self.inner.chat_capability(model, capability)
        }
    }

    fn not_constrained_tool() -> Warning {
        Warning::new(
            WarningKind::Unsupported,
            Subject::Field {
                field: RequestField::Tools,
            },
            "tool `submit_structured_output` is not strict-representable: the subschema at \
             `/properties/contact` is a map; it was sent without `strict`",
        )
    }

    #[tokio::test]
    async fn degradations_name_the_callers_type() {
        let provider = MockProvider::with_response(
            ChatResponseBuilder::new()
                .tool_call(
                    "call_1",
                    EXTRACTION_TOOL,
                    json!({"name": "Ada", "contact": null}),
                )
                .warning(not_constrained_tool())
                .build(),
        );
        let request = request().with_option(ChatExtractionMode::ForcedTool);
        let extracted = provider.extract::<Customer>(&request).await.unwrap();
        assert_eq!(
            extracted.metadata.degradations[0].message,
            "the schema of `Customer` is not strict-representable: the subschema at \
             `Customer.contact` is a map; it was sent without `strict`"
        );
        // The response keeps the adapter's own words.
        assert_eq!(extracted.response.warnings[0], not_constrained_tool());
    }

    #[cfg(feature = "warning-events")]
    #[test]
    fn enforcement_is_a_warning_event_once_per_type_and_model() {
        let (_, events) = crate::warning::capture::captured(async {
            for _ in 0..2 {
                let provider = MockProvider::with_response(approximate_text(
                    r#"{"name": "Ada", "contact": null}"#,
                ))
                .with_provider_name("enforcement-events");
                let request = ChatRequest::new("enforcement-model")
                    .user("Ada")
                    .with_option(ChatExtractionMode::Native);
                provider.extract::<Customer>(&request).await.unwrap();
            }
        });
        let ours: Vec<_> = events
            .iter()
            .filter(|e| e.code.as_deref() == Some("chat.extraction.approximate.Customer"))
            .collect();
        assert_eq!(ours.len(), 2, "{events:?}");
        assert_eq!(ours[0].level, tracing::Level::WARN);
        assert_eq!(ours[1].level, tracing::Level::DEBUG);
        assert!(
            ours[0].message.ends_with(
                "; `Customer` is extracted from `enforcement-model` approximately (`native` mode)"
            ),
            "{}",
            ours[0].message
        );
    }

    #[tokio::test]
    async fn a_floor_fails_before_sending_under_encoding_coverage() {
        let provider = Validating {
            inner: MockProvider::with_text(r#"{"name": "Ada", "contact": null}"#),
            warnings: vec![not_constrained_tool()],
            coverage: crate::ValidationCoverage::Encoding,
        };
        let request = request().with_option(ChatExtractionMode::ForcedTool);
        let err = ChatExtractor::new()
            .min_enforcement(SchemaEnforcement::Approximate)
            .extract::<_, Customer>(&provider, &request)
            .await
            .unwrap_err();
        assert_eq!(provider.inner.call_count(), 0);
        let failure = failure(err);
        let ChatExtractError::EnforcementBelow {
            required,
            achieved,
            reason,
        } = failure.error()
        else {
            panic!("a floor failure expected: {failure:?}");
        };
        assert_eq!(
            (*required, *achieved),
            (
                SchemaEnforcement::Approximate,
                SchemaEnforcement::NotEnforced
            )
        );
        assert!(reason.contains("`Customer.contact` is a map"), "{reason}");
        assert!(failure.response().is_none());
        assert_eq!(failure.metadata().passes, 0);

        // A floor the plan meets lets the call through.
        let extracted = ChatExtractor::new()
            .min_enforcement(SchemaEnforcement::NotEnforced)
            .extract::<_, Customer>(&provider, &request)
            .await;
        assert!(extracted.is_err(), "the mock answers text, not a tool call");
        assert_eq!(provider.inner.call_count(), 1);
    }

    #[tokio::test]
    async fn a_floor_is_checked_after_the_pass_under_capabilities_coverage() {
        let provider = Validating {
            inner: MockProvider::with_response(approximate_text(
                r#"{"name": "Ada", "contact": null}"#,
            )),
            warnings: Vec::new(),
            coverage: crate::ValidationCoverage::Capabilities,
        };
        let request = request().with_option(ChatExtractionMode::Native);
        let err = ChatExtractor::new()
            .min_enforcement(SchemaEnforcement::Exact)
            .extract::<_, Customer>(&provider, &request)
            .await
            .unwrap_err();
        assert_eq!(provider.inner.call_count(), 1);
        let failure = failure(err);
        assert!(
            matches!(
                failure.error(),
                ChatExtractError::EnforcementBelow {
                    achieved: SchemaEnforcement::Approximate,
                    ..
                }
            ),
            "{failure:?}"
        );
        // A paid call: the response and its usage are kept.
        assert!(failure.response().is_some());
        assert_eq!(
            failure
                .metadata()
                .usage
                .as_ref()
                .and_then(|u| u.output_tokens),
            Some(5)
        );
        assert!(!Error::from(failure).is_retryable());
    }

    #[tokio::test]
    async fn a_floor_applies_to_streams() {
        let provider = MockStreamingProvider::from_responses([approximate_text(
            r#"{"name": "Ada", "contact": null}"#,
        )]);
        let request = request().with_option(ChatExtractionMode::Native);
        let mut stream = ChatExtractor::new()
            .min_enforcement(SchemaEnforcement::Exact)
            .extract_stream::<_, Customer>(&provider, &request)
            .await
            .unwrap();
        while stream.next().await.is_some() {}
        let failure = failure(stream.finish().unwrap_err());
        assert!(matches!(
            failure.error(),
            ChatExtractError::EnforcementBelow { .. }
        ));
    }

    #[test]
    fn enforcement_levels_are_ordered() {
        use SchemaEnforcement::{Approximate, Exact, NotEnforced};
        assert!(Exact.satisfies(Exact) && Exact.satisfies(NotEnforced));
        assert!(Approximate.satisfies(Approximate) && !Approximate.satisfies(Exact));
        assert!(NotEnforced.satisfies(NotEnforced) && !NotEnforced.satisfies(Approximate));
        assert_eq!(NotEnforced.to_string(), "not_enforced");
    }

    #[test]
    fn a_plan_reports_mode_enforcement_and_coverage_without_sending() {
        let provider = Validating {
            inner: MockProvider::with_text("unused")
                .with_supported_chat_capabilities(NATIVE_AND_TOOLS),
            warnings: vec![approximated(RequestField::ResponseFormat)],
            coverage: crate::ValidationCoverage::Encoding,
        };
        let plan = ChatExtractor::new()
            .plan::<Customer>(&provider, &request())
            .unwrap();
        assert_eq!(plan.mode, ChatExtractionMode::Native);
        assert_eq!(plan.enforcement, SchemaEnforcement::Approximate);
        assert_eq!(plan.degradations.len(), 1);
        assert_eq!(plan.coverage, crate::ValidationCoverage::Encoding);
        assert_eq!(plan.auto_fallback, None);
        assert_eq!(provider.inner.call_count(), 0);

        // An `Auto` plan follows a refused response format.
        let provider = MockProvider::with_error(refused(RequestField::ResponseFormat))
            .with_supported_chat_capabilities(NATIVE_AND_TOOLS);
        let plan = ChatExtractor::new().plan::<Customer>(&provider, &request());
        // The default validation checks capabilities only: nothing refuses.
        let plan = plan.unwrap();
        assert_eq!(plan.mode, ChatExtractionMode::Native);
        assert_eq!(plan.coverage, crate::ValidationCoverage::Capabilities);
    }

    #[test]
    fn metadata_without_enforcement_fields_deserializes() {
        let metadata: ChatExtractionMetadata =
            serde_json::from_value(json!({"mode": "native", "passes": 1, "repaired": false}))
                .unwrap();
        assert_eq!(metadata.enforcement, SchemaEnforcement::NotEnforced);
        assert!(metadata.degradations.is_empty());

        let mut degraded = ChatExtractionMetadata::new(ChatExtractionMode::ForcedTool);
        degraded.record_pass(
            &ChatResponseBuilder::new()
                .warning(strict_dropped(RequestField::Tools))
                .build(),
            "Person",
        );
        let value = serde_json::to_value(&degraded).unwrap();
        assert_eq!(value["enforcement"], "not_enforced");
        assert_eq!(value["degradations"][0]["subject"]["field"], "tools");
        assert_eq!(
            serde_json::from_value::<ChatExtractionMetadata>(value).unwrap(),
            degraded
        );
    }
}
