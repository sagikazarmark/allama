//! The `/v1/systemone` wire format and its conversion from and to the
//! portable decision types.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::marker::PhantomData;

use allama::adapter::Degradations;
use allama::decision::{
    Answer, ChoiceAnswer, DecisionInput, Question, ScoreAnswer, ScoreLevel, YesNoAnswer,
};
use allama::{
    DecisionRequest, DecisionResponse, Error, ExtraMap, MediaSource, Result, Subject, Usage,
};
use serde::de::{MapAccess, Visitor};
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::value::RawValue;

use crate::Gateway;
use crate::options::{DecisionRequestOptions, KeepAlive};

/// Request body fields that [`DecisionRequestOptions::extra`] must not
/// replace.
const RESERVED: &[&str] = &["model", "state", "questions", "keep_alive", "images"];

/// The System One dialect an endpoint speaks: the protocol is the same, but
/// some servers extend it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Dialect {
    /// TypeSafe's protocol, as its gateways forward it: no images.
    SystemOne,
    /// Ollama's: `keep_alive`, and images as bare base64.
    Ollama,
    /// Cloudflare Workers AI's (Clef models): the model's own URL, a model
    /// selector in the body, and images with their media type.
    ///
    /// What each model's input schema allows (its limits, image types and
    /// question names) is the model's to enforce (ADR 0015): it answers
    /// what it refuses with an error that names the field.
    WorkersAi,
}

/// A `/v1/systemone` request body.
#[derive(Debug, Serialize)]
pub(crate) struct WireRequest<'a> {
    model: &'a str,
    state: &'a DecisionInput,
    #[serde(serialize_with = "serialize_entries")]
    questions: Vec<(&'a str, WireQuestion<'a>)>,
    #[serde(skip_serializing_if = "Option::is_none")]
    keep_alive: Option<&'a KeepAlive>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    images: Vec<WireImage<'a>>,
    #[serde(flatten)]
    extra: &'a ExtraMap,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum WireQuestion<'a> {
    Noul {
        instructions: &'a str,
        #[serde(skip_serializing_if = "NoulCriteria::is_empty")]
        criteria: NoulCriteria<'a>,
    },
    Choice {
        instructions: &'a str,
        #[serde(serialize_with = "serialize_entries")]
        criteria: Vec<(&'a str, Option<&'a str>)>,
    },
    Score {
        instructions: &'a str,
        criteria: Vec<Cow<'a, str>>,
    },
}

#[derive(Debug, Serialize)]
struct NoulCriteria<'a> {
    #[serde(rename = "false", skip_serializing_if = "Option::is_none")]
    no: Option<&'a str>,
    #[serde(rename = "true", skip_serializing_if = "Option::is_none")]
    yes: Option<&'a str>,
}

impl NoulCriteria<'_> {
    fn is_empty(&self) -> bool {
        self.no.is_none() && self.yes.is_none()
    }
}

/// An image as a dialect takes it.
#[derive(Debug, Serialize)]
#[serde(untagged)]
enum WireImage<'a> {
    /// Ollama: the base64 data alone.
    Bare(&'a str),
    /// Workers AI: the data with its media type.
    Typed {
        content_type: &'a str,
        base64: &'a str,
    },
}

/// Encode a portable request for `dialect`, reporting through `d`. The
/// request must already be validated.
pub(crate) fn encode<'a>(
    request: &'a DecisionRequest,
    options: &'a DecisionRequestOptions,
    dialect: Dialect,
    d: &mut Degradations,
) -> Result<WireRequest<'a>> {
    encode_parts(
        request,
        options.keep_alive.as_ref(),
        &options.extra,
        dialect,
        d,
    )
}

/// [`encode`] from the options' parts.
pub(crate) fn encode_parts<'a>(
    request: &'a DecisionRequest,
    keep_alive: Option<&'a KeepAlive>,
    extra: &'a ExtraMap,
    dialect: Dialect,
    d: &mut Degradations,
) -> Result<WireRequest<'a>> {
    // Go's `encoding/json` (Ollama) matches keys case-insensitively.
    if let Some(key) = extra.keys().find(|key| {
        RESERVED
            .iter()
            .any(|reserved| key.eq_ignore_ascii_case(reserved))
    }) {
        return Err(Error::invalid_request(format!(
            "`extra` cannot set the request field {key:?}; use the portable request or the typed options"
        )).authored());
    }
    let questions = request
        .questions
        .iter()
        .map(|(name, question)| Ok((name.as_str(), encode_question(name, question)?)))
        .collect::<Result<_>>()?;
    Ok(WireRequest {
        model: match dialect {
            Dialect::WorkersAi => workers_ai_selector(&request.model),
            Dialect::SystemOne | Dialect::Ollama => &request.model,
        },
        state: &request.input,
        questions,
        keep_alive,
        images: encode_images(request, dialect, d)?,
        extra,
    })
}

/// The model selector Workers AI reads from the body: the last segment of
/// the model's name (`clef` for `@cf/cloudflare/clef`), which also names
/// the model in the URL.
pub(crate) fn workers_ai_selector(model: &str) -> &str {
    model.rsplit('/').next().unwrap_or(model)
}

/// The request's images as `dialect` takes them: inline data only, since
/// no System One server downloads a URL. An image detail is a hint none of
/// them reads.
fn encode_images<'a>(
    request: &'a DecisionRequest,
    dialect: Dialect,
    d: &mut Degradations,
) -> Result<Vec<WireImage<'a>>> {
    if dialect == Dialect::SystemOne && !request.images.is_empty() {
        return Err(d.reject(
            Subject::Other {
                name: "images".into(),
            },
            "this System One endpoint takes no images",
        ));
    }
    let mut images = Vec::with_capacity(request.images.len());
    for (index, image) in request.images.iter().enumerate() {
        let subject = || Subject::Other {
            name: format!("image {index}"),
        };
        let MediaSource::Base64 {
            media_type, data, ..
        } = &image.source
        else {
            return Err(d.reject(
                subject(),
                "System One takes images as inline data only: send the image's bytes (`Image::bytes`), not a URL or a provider file",
            ));
        };
        if image.detail.is_some() {
            d.hint(
                subject(),
                "System One has no image detail setting; the image was sent without it",
            )?;
        }
        images.push(match dialect {
            Dialect::SystemOne | Dialect::Ollama => WireImage::Bare(data),
            Dialect::WorkersAi => WireImage::Typed {
                content_type: image.resolved_media_type()?.unwrap_or(media_type),
                base64: data,
            },
        });
    }
    Ok(images)
}

fn encode_question<'a>(name: &str, question: &'a Question) -> Result<WireQuestion<'a>> {
    Ok(match question {
        Question::YesNo(question) => WireQuestion::Noul {
            instructions: &question.instructions,
            criteria: NoulCriteria {
                no: question.no.as_deref(),
                yes: question.yes.as_deref(),
            },
        },
        Question::Choice(question) => WireQuestion::Choice {
            instructions: &question.instructions,
            criteria: question
                .options
                .iter()
                .map(|option| (option.key.as_str(), option.description.as_deref()))
                .collect(),
        },
        Question::Score(question) => {
            let criteria: Vec<_> = question.levels.iter().map(level_criterion).collect();
            let mut seen: HashSet<&str> = HashSet::new();
            if let Some(repeated) = criteria
                .iter()
                .map(AsRef::as_ref)
                .find(|criterion| !seen.insert(criterion))
            {
                return Err(Error::invalid_request(format!(
                    "question {name:?}: two levels both read {repeated:?} on the System One wire"
                )));
            }
            WireQuestion::Score {
                instructions: &question.instructions,
                criteria,
            }
        }
        other => {
            return Err(Error::unsupported(format!(
                "System One has no {} questions (question {name:?})",
                other.kind()
            ))
            .authored());
        }
    })
}

/// System One describes a level with one string: `label: description`, the
/// convention of its own examples, or the label alone.
fn level_criterion(level: &ScoreLevel) -> Cow<'_, str> {
    match &level.description {
        Some(description) => Cow::Owned(format!("{}: {description}", level.label)),
        None => Cow::Borrowed(&level.label),
    }
}

/// The Cloudflare API's envelope around a Workers AI model's output:
/// `{"result": .., "success": .., "errors": [..], "messages": [..]}`. Its
/// errors are read as an error envelope.
#[derive(Debug, Deserialize)]
pub(crate) struct CloudflareEnvelope {
    pub(crate) result: Box<RawValue>,
}

/// A `/v1/systemone` response body. Answers stay raw until their question
/// is known, so a malformed answer is reported by name.
#[derive(Debug, Deserialize)]
pub(crate) struct WireResponse {
    #[serde(default)]
    model: Option<String>,
    #[serde(deserialize_with = "deserialize_entries")]
    answers: Vec<(String, Box<RawValue>)>,
    #[serde(default)]
    usage: Option<WireUsage>,
    /// Response id (OpenRouter).
    #[serde(default)]
    id: Option<String>,
    /// Gateway addition (OpenRouter).
    #[serde(default)]
    provider: Option<String>,
}

#[derive(Debug, Deserialize)]
struct WireUsage {
    #[serde(default)]
    input_tokens: Option<u64>,
    #[serde(default)]
    output_tokens: Option<u64>,
    /// Gateway addition (OpenRouter).
    #[serde(default)]
    cost: Option<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum WireAnswer {
    Noul {
        noul: f64,
    },
    Choice {
        choice: String,
        #[serde(deserialize_with = "deserialize_entries")]
        probabilities: Vec<(String, f64)>,
        #[serde(default)]
        confidence: Option<f64>,
    },
    Score {
        score: f64,
        #[serde(deserialize_with = "deserialize_entries")]
        probabilities: Vec<(String, f64)>,
        #[serde(default)]
        confidence: Option<f64>,
    },
}

/// The decoded answers, each question's confidence statistic when the
/// provider reported one, and gateway metadata.
pub(crate) struct Decoded {
    pub(crate) response: DecisionResponse,
    pub(crate) confidence: BTreeMap<String, f64>,
    pub(crate) gateway: Option<Gateway>,
}

/// Decode a response body into portable answers.
pub(crate) fn decode(body: WireResponse) -> Result<Decoded> {
    let mut confidence = BTreeMap::new();
    let mut answers = Vec::with_capacity(body.answers.len());
    for (name, raw) in body.answers {
        let wire: WireAnswer = serde_json::from_str(raw.get()).map_err(|err| {
            Error::unexpected_response(format!("decision question {name:?}: {err}"))
        })?;
        let answer = match wire {
            WireAnswer::Noul { noul } => Answer::from(YesNoAnswer::new(noul)),
            WireAnswer::Choice {
                choice,
                probabilities,
                confidence: reported,
            } => {
                confidence.extend(reported.map(|c| (name.clone(), c)));
                ChoiceAnswer::new(choice, probabilities).into()
            }
            WireAnswer::Score {
                score,
                probabilities,
                confidence: reported,
            } => {
                confidence.extend(reported.map(|c| (name.clone(), c)));
                ScoreAnswer::new(score, levels(&name, probabilities)?).into()
            }
        };
        answers.push((name, answer));
    }
    let cost = body.usage.as_ref().and_then(|usage| usage.cost);
    let gateway = (body.provider.is_some() || cost.is_some()).then_some(Gateway {
        provider: body.provider,
        cost,
    });
    let mut response = DecisionResponse::new(answers);
    response.id = body.id;
    response.model = body.model;
    response.usage = body.usage.map(|usage| {
        let mut converted = Usage::new();
        converted.input_tokens = usage.input_tokens;
        converted.output_tokens = usage.output_tokens;
        converted.with_derived_total()
    });
    Ok(Decoded {
        response,
        confidence,
        gateway,
    })
}

/// Turn `{"0": p0, "1": p1, ...}` into `[p0, p1, ...]`.
fn levels(name: &str, probabilities: Vec<(String, f64)>) -> Result<Vec<f64>> {
    let fail = |problem: String| {
        Error::unexpected_response(format!("decision question {name:?} {problem}"))
    };
    let mut by_level = BTreeMap::new();
    for (key, probability) in probabilities {
        let level = key
            .parse::<usize>()
            .map_err(|_| fail(format!("reports an unexpected score level {key:?}")))?;
        if by_level.insert(level, probability).is_some() {
            return Err(fail(format!("reports score level {level} more than once")));
        }
    }
    let present: HashSet<usize> = by_level.keys().copied().collect();
    if let Some(missing) = (0..by_level.len()).find(|level| !present.contains(level)) {
        return Err(fail(format!(
            "reports no probability for score level {missing}"
        )));
    }
    Ok(by_level.into_values().collect())
}

fn serialize_entries<S, K, V>(
    entries: &[(K, V)],
    serializer: S,
) -> std::result::Result<S::Ok, S::Error>
where
    S: Serializer,
    K: Serialize,
    V: Serialize,
{
    let mut map = serializer.serialize_map(Some(entries.len()))?;
    for (key, value) in entries {
        map.serialize_entry(key, value)?;
    }
    map.end()
}

/// Deserialize an object into its entries, in document order.
fn deserialize_entries<'de, D, V>(
    deserializer: D,
) -> std::result::Result<Vec<(String, V)>, D::Error>
where
    D: Deserializer<'de>,
    V: Deserialize<'de>,
{
    struct Entries<V>(PhantomData<V>);

    impl<'de, V: Deserialize<'de>> Visitor<'de> for Entries<V> {
        type Value = Vec<(String, V)>;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("a map")
        }

        fn visit_map<A: MapAccess<'de>>(
            self,
            mut access: A,
        ) -> std::result::Result<Self::Value, A::Error> {
            let mut entries = Vec::with_capacity(cautious_capacity(access.size_hint()));
            while let Some(entry) = access.next_entry()? {
                entries.push(entry);
            }
            Ok(entries)
        }
    }

    deserializer.deserialize_map(Entries(PhantomData))
}

/// A capacity to preallocate from a deserializer's `size_hint`, capped:
/// a length-prefixed format reports the length the input declares,
/// so trusting it lets one small message ask for terabytes. The
/// same cap as `allama`'s decision decoders, and serde's own collections.
fn cautious_capacity(hint: Option<usize>) -> usize {
    const MAX_PREALLOCATED: usize = 4096;
    hint.unwrap_or(0).min(MAX_PREALLOCATED)
}

#[cfg(test)]
mod tests {
    use super::*;
    use allama::{Image, ImageDetail, UnsupportedPolicy, Warning};
    use serde_json::json;

    /// Encode as `dialect` speaks, with the warnings it reports.
    fn encode_as(
        request: &DecisionRequest,
        options: &DecisionRequestOptions,
        dialect: Dialect,
    ) -> Result<(serde_json::Value, Vec<Warning>)> {
        let mut d = Degradations::new("test", UnsupportedPolicy::Warn);
        let body = serde_json::to_value(encode(request, options, dialect, &mut d)?)?;
        Ok((body, d.into_warnings()))
    }

    fn encode_one(request: &DecisionRequest, options: &DecisionRequestOptions) -> Result<String> {
        let mut d = Degradations::new("test", UnsupportedPolicy::Warn);
        Ok(serde_json::to_string(&encode(
            request,
            options,
            Dialect::SystemOne,
            &mut d,
        )?)?)
    }

    /// Regression: an object whose declared length is
    /// absurd preallocated that much, a capacity overflow (a panic) for
    /// `usize::MAX`.
    #[test]
    fn a_declared_length_is_not_trusted() {
        struct Lying<I>(I);
        impl<I: Iterator> Iterator for Lying<I> {
            type Item = I::Item;
            fn next(&mut self) -> Option<I::Item> {
                self.0.next()
            }
            fn size_hint(&self) -> (usize, Option<usize>) {
                (usize::MAX, Some(usize::MAX))
            }
        }
        let map = serde::de::value::MapDeserializer::<_, serde::de::value::Error>::new(Lying(
            [("a", 1u32)].into_iter(),
        ));
        let entries: Vec<(String, u32)> = deserialize_entries(map).unwrap();
        assert_eq!(entries, [("a".to_owned(), 1)]);
    }

    fn request() -> DecisionRequest {
        DecisionRequest::new("nimble", json!({"ticket": "I was charged twice."}))
            .question(
                "label",
                Question::choice("Which label fits this ticket?")
                    .option("billing", "Payments and refunds")
                    .bare_option("bug")
                    .option("account", "Login and account access"),
            )
            .question(
                "refund",
                Question::yes_no("Is the customer requesting a refund?")
                    .no("No refund is requested")
                    .yes("The customer requests a refund"),
            )
            .question("spam", Question::yes_no("Is this spam?"))
            .question(
                "urgency",
                Question::score("How urgently does this ticket need a response?")
                    .level("Routine", "no time pressure")
                    .bare_level("Soon")
                    .bare_level("Immediate"),
            )
    }

    #[test]
    fn encodes_questions_in_order_with_wire_names() {
        let options = DecisionRequestOptions::default();
        let body = encode_one(&request(), &options).unwrap();
        assert_eq!(
            body,
            concat!(
                r#"{"model":"nimble","state":{"ticket":"I was charged twice."},"questions":{"#,
                r#""label":{"type":"choice","instructions":"Which label fits this ticket?","criteria":{"billing":"Payments and refunds","bug":null,"account":"Login and account access"}},"#,
                r#""refund":{"type":"noul","instructions":"Is the customer requesting a refund?","criteria":{"false":"No refund is requested","true":"The customer requests a refund"}},"#,
                r#""spam":{"type":"noul","instructions":"Is this spam?"},"#,
                r#""urgency":{"type":"score","instructions":"How urgently does this ticket need a response?","criteria":["Routine: no time pressure","Soon","Immediate"]}}}"#,
            )
        );
    }

    #[test]
    fn encodes_options() {
        let options = DecisionRequestOptions {
            keep_alive: Some(KeepAlive::from("5m")),
            extra: [("seed".to_owned(), json!(1))].into_iter().collect(),
            ..DecisionRequestOptions::default()
        };
        let (body, _) = encode_as(&request(), &options, Dialect::Ollama).unwrap();
        assert_eq!(body["keep_alive"], json!("5m"));
        assert_eq!(body["seed"], json!(1));
        assert!(body.get("images").is_none());
    }

    #[test]
    fn images_take_each_dialects_form() {
        let request = request()
            .image(Image::base64("image/png", "aGk="))
            .image(Image::base64("image/jpeg", "aG8="));
        let options = DecisionRequestOptions::default();
        let (ollama, warnings) = encode_as(&request, &options, Dialect::Ollama).unwrap();
        assert_eq!(ollama["images"], json!(["aGk=", "aG8="]));
        assert!(warnings.is_empty(), "{warnings:?}");
        let (workers_ai, _) = encode_as(&request, &options, Dialect::WorkersAi).unwrap();
        assert_eq!(
            workers_ai["images"],
            json!([
                {"content_type": "image/png", "base64": "aGk="},
                {"content_type": "image/jpeg", "base64": "aG8="}
            ])
        );
        let err = encode_as(&request, &options, Dialect::SystemOne).unwrap_err();
        assert!(matches!(err, Error::Unsupported { .. }), "{err:?}");
    }

    #[test]
    fn images_are_inline_data_only() {
        let options = DecisionRequestOptions::default();
        for image in [
            Image::url("https://example.com/a.png"),
            Image::provider_file("openai", "file-1"),
        ] {
            for dialect in [Dialect::Ollama, Dialect::WorkersAi] {
                let err =
                    encode_as(&request().image(image.clone()), &options, dialect).unwrap_err();
                assert!(
                    matches!(err, Error::Unsupported { .. }),
                    "{dialect:?}: {err:?}"
                );
            }
        }
        // The media type is the model's to judge (ADR 0015).
        let gif = request().image(Image::base64("image/gif", "aGk="));
        let (body, _) = encode_as(&gif, &options, Dialect::WorkersAi).unwrap();
        assert_eq!(body["images"][0]["content_type"], "image/gif");
    }

    #[test]
    fn an_image_detail_is_dropped_with_a_warning() {
        let request =
            request().image(Image::base64("image/webp", "aGk=").with_detail(ImageDetail::High));
        let (body, warnings) = encode_as(
            &request,
            &DecisionRequestOptions::default(),
            Dialect::WorkersAi,
        )
        .unwrap();
        assert_eq!(body["images"][0]["content_type"], "image/webp");
        assert_eq!(warnings.len(), 1, "{warnings:?}");
    }

    #[test]
    fn workers_ai_selects_the_model_by_its_last_segment() {
        let request = DecisionRequest::new("@cf/cloudflare/clef-flash", "x")
            .question("q", Question::yes_no("?"));
        let options = DecisionRequestOptions::default();
        let (body, _) = encode_as(&request, &options, Dialect::WorkersAi).unwrap();
        assert_eq!(body["model"], "clef-flash");
        let (body, _) = encode_as(&request, &options, Dialect::SystemOne).unwrap();
        assert_eq!(body["model"], "@cf/cloudflare/clef-flash");
    }

    #[test]
    fn levels_must_stay_distinct_on_the_wire() {
        let request = DecisionRequest::new("m", "x")
            .question("q", Question::score("?").level("x", "y").bare_level("x: y"));
        let err = encode_one(&request, &DecisionRequestOptions::default()).unwrap_err();
        assert!(matches!(err, Error::InvalidRequest { .. }), "{err:?}");
    }

    #[test]
    fn extra_cannot_replace_request_fields() {
        for key in RESERVED
            .iter()
            .copied()
            .chain(["Model", "STATE", "Keep_Alive"])
        {
            let options = DecisionRequestOptions {
                extra: [(key.to_owned(), json!("other"))].into_iter().collect(),
                ..DecisionRequestOptions::default()
            };
            let err = encode_one(&request(), &options).unwrap_err();
            assert!(
                matches!(err, Error::InvalidRequest { .. }),
                "{key}: {err:?}"
            );
        }
    }

    #[test]
    fn decodes_answers_usage_confidence_and_gateway_fields() {
        let body: WireResponse = serde_json::from_str(
            r#"{
            "id": "gen-1",
            "provider": "TypeSafe",
            "model": "nimble",
            "answers": {
                "label": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.9781, "bug": 0.0125, "account": 0.0093}, "confidence": 0.8906},
                "refund": {"type": "noul", "noul": 0.9989},
                "urgency": {"type": "score", "score": 0.8308, "legend": {"0": "Routine", "1": "Soon", "2": "Immediate"}, "probabilities": {"0": 0.3, "1": 0.5692, "2": 0.1308}, "confidence": 0.1371}
            },
            "usage": {"input_tokens": 174, "output_tokens": 3, "cost": 0.00001}
        }"#,
        )
        .unwrap();
        let Decoded {
            response,
            confidence,
            gateway,
        } = decode(body).unwrap();
        assert_eq!(response.model.as_deref(), Some("nimble"));
        let label = response.choice("label").unwrap();
        assert_eq!(label.chosen, "billing");
        assert_eq!(label.probabilities[1], ("bug".to_owned(), 0.0125));
        assert_eq!(response.yes_no("refund").unwrap().probability, 0.9989);
        assert_eq!(
            response.score("urgency").unwrap().probabilities,
            [0.3, 0.5692, 0.1308]
        );
        let usage = response.usage.unwrap();
        assert_eq!(usage.total_tokens, Some(177));
        assert_eq!(confidence["label"], 0.8906);
        assert_eq!(confidence["urgency"], 0.1371);
        assert!(!confidence.contains_key("refund"));
        assert_eq!(response.id.as_deref(), Some("gen-1"));
        let gateway = gateway.unwrap();
        assert_eq!(gateway.provider.as_deref(), Some("TypeSafe"));
        assert_eq!(gateway.cost, Some(0.00001));
    }

    #[test]
    fn malformed_answers_name_the_question() {
        let decode_one = |answer: serde_json::Value| {
            let body: WireResponse =
                serde_json::from_value(json!({"answers": {"q": answer}})).unwrap();
            match decode(body) {
                Err(Error::UnexpectedResponse { message, .. }) => message,
                Err(other) => panic!("unexpected error {other:?}"),
                Ok(_) => panic!("expected an error"),
            }
        };
        assert!(decode_one(json!({"type": "verdict", "x": 1})).contains("\"q\""));
        assert!(
            decode_one(json!({"type": "score", "score": 1, "probabilities": {"0": 0.5, "2": 0.5}}))
                .contains("no probability for score level 1")
        );
        assert!(
            decode_one(json!({"type": "score", "score": 1, "probabilities": {"0": 0.5, "x": 0.5}}))
                .contains("unexpected score level \"x\"")
        );
    }
}
