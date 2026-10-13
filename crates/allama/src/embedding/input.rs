use std::fmt;

use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::macros::text_conversions;
use crate::{Image, ProviderData, ProviderDataEntry, Result, Text};

/// One item to embed: an ordered list of parts the model combines into one
/// vector.
///
/// A request holds one input per vector it returns. Most inputs are a single
/// text and convert from strings, so `.inputs(["a", "b"])` embeds two texts.
/// Several parts in one input are combined into ONE vector:
/// `EmbeddingInput::from(vec![part_a, part_b])` is one input, not two. Pass
/// separate inputs to [`EmbeddingRequest::inputs`](crate::EmbeddingRequest::inputs)
/// to embed parts separately. How a model combines parts, and which
/// combinations it accepts, is model-defined; a model that embeds one part per
/// input rejects several parts with [`Error::Unsupported`](crate::Error::Unsupported).
///
/// # Provider data
///
/// [`provider_data`](Self::provider_data) carries caller-set settings for one
/// provider's per-input wire entry (for example Gemini's document title). The
/// namespace names the provider that reads it; other providers ignore it. Part
/// settings (for example a per-text truncation mode) go in the part's own
/// provider data instead.
///
/// # Serialization
///
/// An input that is exactly one text part without citations or provider data
/// serializes as a bare string; anything else as
/// `{"parts": [..], "provider_data"?: {..}}`, with parts in the same flat shape
/// as chat's [`ContentPart`](crate::ContentPart). A bare string deserializes as
/// a text input. Normalization is semantic, not byte-preserving: an object
/// holding one plain text part re-serializes as a string. Unknown object keys
/// are rejected, so misspelled caller data is never lost silently.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct EmbeddingInput {
    /// Parts combined into one vector, in order.
    pub parts: Vec<EmbeddingPart>,
    /// Caller-set data for one provider's per-input wire entry.
    pub provider_data: ProviderData,
}

/// A part of an [`EmbeddingInput`].
///
/// Unlike [`ContentPart`](crate::ContentPart), which keeps content it does not
/// recognize as `Other`, an embedding part is strict: an unknown `type` fails
/// to deserialize, because dropping a part would silently change the vector.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum EmbeddingPart {
    /// Text.
    ///
    /// [`Text::citations`] describe model output (ADR 0007) and are ignored on
    /// embedding input. [`EmbeddingPart::text`], [`EmbeddingInput::text`], and
    /// the `From` conversions clear them; a part built directly with
    /// `EmbeddingPart::Text(text)` keeps them, and they then count for
    /// equality, hashing, and serialization. Build parts through the
    /// constructors when inputs are used as cache keys.
    Text(Text),
    /// An image, for models that report
    /// [`EmbeddingCapability::ImageInput`](crate::EmbeddingCapability::ImageInput).
    ///
    /// [`Image::detail`] is a hint no embedding API honors: adapters warn, or
    /// fail under [`UnsupportedPolicy::Error`](crate::UnsupportedPolicy::Error).
    Image(Image),
}

impl EmbeddingInput {
    /// A single-part text input.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self::from_parts([EmbeddingPart::text(text)])
    }

    /// A single-part input of an image at `url`.
    #[must_use]
    pub fn image_url(url: impl Into<String>) -> Self {
        Self::from_parts([EmbeddingPart::image_url(url)])
    }

    /// A single-part input of inline base64 image data.
    #[must_use]
    pub fn image_base64(media_type: impl Into<String>, data: impl Into<crate::Base64Data>) -> Self {
        Self::from_parts([EmbeddingPart::image_base64(media_type, data)])
    }

    /// A single-part input of an image from raw bytes, sent inline
    /// (base64-encoded).
    #[must_use]
    pub fn image_bytes(media_type: impl Into<String>, data: impl AsRef<[u8]>) -> Self {
        Self::from_parts([EmbeddingPart::image_bytes(media_type, data)])
    }

    /// An input from parts, combined into one vector.
    #[must_use]
    pub fn from_parts(parts: impl IntoIterator<Item = EmbeddingPart>) -> Self {
        Self {
            parts: parts.into_iter().collect(),
            provider_data: ProviderData::new(),
        }
    }

    /// Replace the provider data.
    #[must_use]
    pub fn with_provider_data(mut self, provider_data: ProviderData) -> Self {
        self.provider_data = provider_data;
        self
    }

    /// Attach a typed provider data entry.
    ///
    /// # Errors
    ///
    /// Returns an error when `entry` fails to serialize.
    pub fn with_provider_entry<T: ProviderDataEntry>(mut self, entry: &T) -> Result<Self> {
        self.provider_data.insert(entry)?;
        Ok(self)
    }

    /// The parts.
    #[must_use]
    pub fn parts(&self) -> &[EmbeddingPart] {
        &self.parts
    }

    /// The text when this input is exactly one text part.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self.parts.as_slice() {
            [part] => part.as_text(),
            _ => None,
        }
    }

    /// The text when this input serializes as a bare string.
    fn as_plain_text(&self) -> Option<&str> {
        match self.parts.as_slice() {
            [EmbeddingPart::Text(text)]
                if self.provider_data.is_empty()
                    && text.citations.is_empty()
                    && text.provider_data.is_empty() =>
            {
                Some(&text.text)
            }
            _ => None,
        }
    }
}

impl EmbeddingPart {
    /// A text part.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(Text::new(text))
    }

    /// An image part from a URL.
    #[must_use]
    pub fn image_url(url: impl Into<String>) -> Self {
        Self::Image(Image::url(url))
    }

    /// An image part from inline base64 data.
    #[must_use]
    pub fn image_base64(media_type: impl Into<String>, data: impl Into<crate::Base64Data>) -> Self {
        Self::Image(Image::base64(media_type, data))
    }

    /// An image part from raw bytes, sent inline (base64-encoded).
    #[must_use]
    pub fn image_bytes(media_type: impl Into<String>, data: impl AsRef<[u8]>) -> Self {
        Self::Image(Image::bytes(media_type, data))
    }

    /// The text of a text part.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(&text.text),
            Self::Image(_) => None,
        }
    }

    /// The image of an image part.
    #[must_use]
    pub fn as_image(&self) -> Option<&Image> {
        match self {
            Self::Image(image) => Some(image),
            Self::Text(_) => None,
        }
    }

    /// Stable snake_case name of the part kind, for messages.
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Text(_) => "text",
            Self::Image(_) => "image",
        }
    }

    /// The part's provider data.
    #[must_use]
    pub fn provider_data(&self) -> &ProviderData {
        match self {
            Self::Text(text) => &text.provider_data,
            Self::Image(image) => &image.provider_data,
        }
    }
}

impl From<Image> for EmbeddingPart {
    fn from(image: Image) -> Self {
        Self::Image(image)
    }
}

impl From<Image> for EmbeddingInput {
    fn from(image: Image) -> Self {
        Self::from_parts([EmbeddingPart::Image(image)])
    }
}

impl From<Text> for EmbeddingPart {
    /// Clears citations, which describe model output.
    fn from(mut text: Text) -> Self {
        text.citations.clear();
        Self::Text(text)
    }
}

text_conversions!(EmbeddingInput, Self::text);

impl From<Text> for EmbeddingInput {
    /// Clears citations, which describe model output.
    fn from(text: Text) -> Self {
        Self::from_parts([EmbeddingPart::from(text)])
    }
}

impl From<EmbeddingPart> for EmbeddingInput {
    fn from(part: EmbeddingPart) -> Self {
        Self::from_parts([part])
    }
}

impl From<Vec<EmbeddingPart>> for EmbeddingInput {
    /// One input combining every part into a single vector.
    fn from(parts: Vec<EmbeddingPart>) -> Self {
        Self::from_parts(parts)
    }
}

#[derive(Serialize)]
struct InputRef<'a> {
    parts: &'a [EmbeddingPart],
    #[serde(skip_serializing_if = "ProviderData::is_empty")]
    provider_data: &'a ProviderData,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InputObject {
    parts: Vec<EmbeddingPart>,
    #[serde(default)]
    provider_data: ProviderData,
}

impl Serialize for EmbeddingInput {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        if let Some(text) = self.as_plain_text() {
            return serializer.serialize_str(text);
        }
        InputRef {
            parts: &self.parts,
            provider_data: &self.provider_data,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for EmbeddingInput {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        deserializer.deserialize_any(InputVisitor)
    }
}

struct InputVisitor;

impl<'de> Visitor<'de> for InputVisitor {
    type Value = EmbeddingInput;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a string or an object with `parts`")
    }

    fn visit_str<E: de::Error>(self, text: &str) -> std::result::Result<Self::Value, E> {
        Ok(EmbeddingInput::text(text))
    }

    fn visit_string<E: de::Error>(self, text: String) -> std::result::Result<Self::Value, E> {
        Ok(EmbeddingInput::text(text))
    }

    fn visit_map<A: MapAccess<'de>>(self, map: A) -> std::result::Result<Self::Value, A::Error> {
        let InputObject {
            parts,
            provider_data,
        } = InputObject::deserialize(de::value::MapAccessDeserializer::new(map))?;
        Ok(EmbeddingInput {
            parts,
            provider_data,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::hash::{DefaultHasher, Hash, Hasher};

    use serde_json::json;

    use super::*;
    use crate::{Citation, CitationSource, ContentPart};

    fn hash<T: Hash>(value: &T) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    fn cited(text: &str) -> Text {
        Text::new(text).with_citations(vec![Citation::new(CitationSource::url(
            "https://example.com",
            None,
        ))])
    }

    #[test]
    fn every_string_form_converts_to_a_text_input() {
        let owned = String::from("a");
        let expected = EmbeddingInput::text("a");
        assert_eq!(EmbeddingInput::from("a"), expected);
        assert_eq!(EmbeddingInput::from(owned.clone()), expected);
        assert_eq!(EmbeddingInput::from(&owned), expected);
        assert_eq!(EmbeddingInput::from(Box::<str>::from("a")), expected);
        assert_eq!(EmbeddingInput::from(Cow::Borrowed("a")), expected);
        assert_eq!(EmbeddingInput::from(Text::new("a")), expected);
        assert_eq!(EmbeddingInput::from(EmbeddingPart::text("a")), expected);
        assert_eq!(
            EmbeddingInput::from(vec![EmbeddingPart::text("a")]),
            expected
        );
    }

    #[test]
    fn as_text_requires_exactly_one_text_part() {
        assert_eq!(EmbeddingInput::text("a").as_text(), Some("a"));
        assert_eq!(EmbeddingInput::from_parts([]).as_text(), None);
        let two = EmbeddingInput::from(vec![EmbeddingPart::text("a"), EmbeddingPart::text("b")]);
        assert_eq!(two.as_text(), None);
        assert_eq!(two.parts().len(), 2);
    }

    #[test]
    fn conversions_clear_citations() {
        let EmbeddingPart::Text(text) = EmbeddingPart::from(cited("a")) else {
            panic!("expected a text part");
        };
        assert!(text.citations.is_empty());
        let input = EmbeddingInput::from(cited("a"));
        assert_eq!(input, EmbeddingInput::text("a"));
        assert_eq!(hash(&input), hash(&EmbeddingInput::text("a")));
        assert_eq!(serde_json::to_value(&input).unwrap(), json!("a"));
    }

    #[test]
    fn directly_built_parts_keep_citations() {
        let input = EmbeddingInput::from_parts([EmbeddingPart::Text(cited("a"))]);
        assert_ne!(input, EmbeddingInput::text("a"));
        let value = serde_json::to_value(&input).unwrap();
        assert!(value.is_object());
        assert_eq!(
            serde_json::from_value::<EmbeddingInput>(value).unwrap(),
            input
        );
    }

    #[test]
    fn single_text_serializes_as_bare_string() {
        let input = EmbeddingInput::from("hello");
        assert_eq!(serde_json::to_value(&input).unwrap(), json!("hello"));
        assert_eq!(
            serde_json::from_value::<EmbeddingInput>(json!("hello")).unwrap(),
            input
        );
        assert_eq!(
            serde_json::from_str::<EmbeddingInput>(r#""hello""#).unwrap(),
            input
        );
    }

    #[test]
    fn parts_use_the_content_part_shape() {
        let part = EmbeddingPart::text("a");
        let value = serde_json::to_value(&part).unwrap();
        assert_eq!(value, json!({"type": "text", "text": "a"}));
        assert_eq!(serde_json::to_value(ContentPart::text("a")).unwrap(), value);
        assert_eq!(
            serde_json::from_value::<EmbeddingPart>(value).unwrap(),
            part
        );
    }

    #[test]
    fn several_parts_or_provider_data_serialize_as_object() {
        let input = EmbeddingInput::from(vec![EmbeddingPart::text("a"), EmbeddingPart::text("b")]);
        let value = serde_json::to_value(&input).unwrap();
        assert_eq!(
            value,
            json!({"parts": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]})
        );
        assert_eq!(
            serde_json::from_value::<EmbeddingInput>(value).unwrap(),
            input
        );

        let input = EmbeddingInput::text("doc").with_provider_data(ProviderData::new().with_raw(
            "google-ai",
            "title",
            json!("Title"),
        ));
        let value = serde_json::to_value(&input).unwrap();
        assert_eq!(
            value,
            json!({
                "parts": [{"type": "text", "text": "doc"}],
                "provider_data": {"google-ai": {"title": "Title"}}
            })
        );
        assert_eq!(
            serde_json::from_value::<EmbeddingInput>(value).unwrap(),
            input
        );

        let input = EmbeddingInput::from(Text::new("x").with_provider_data(
            ProviderData::new().with_raw("bedrock", "truncationMode", json!("END")),
        ));
        let value = serde_json::to_value(&input).unwrap();
        assert_eq!(
            value,
            json!({"parts": [{
                "type": "text",
                "text": "x",
                "provider_data": {"bedrock": {"truncationMode": "END"}}
            }]})
        );
        assert_eq!(
            serde_json::from_value::<EmbeddingInput>(value).unwrap(),
            input
        );
    }

    #[test]
    fn single_plain_part_object_normalizes_to_bare_string() {
        let input: EmbeddingInput =
            serde_json::from_value(json!({"parts": [{"type": "text", "text": "x"}]})).unwrap();
        assert_eq!(serde_json::to_value(&input).unwrap(), json!("x"));
    }

    #[test]
    fn empty_provider_namespace_is_absent() {
        let mut data = ProviderData::new();
        data.namespace_mut("google-ai");
        let input = EmbeddingInput::text("x").with_provider_data(data);
        assert_eq!(serde_json::to_string(&input).unwrap(), "\"x\"");
        assert_eq!(input, EmbeddingInput::text("x"));
        assert_eq!(hash(&input), hash(&EmbeddingInput::text("x")));
    }

    #[test]
    fn malformed_inputs_report_precise_errors() {
        let err = serde_json::from_str::<EmbeddingInput>(
            r#"{"parts": [{"type": "audio", "data": "x"}]}"#,
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .starts_with("unknown variant `audio`, expected `text`"),
            "{err}"
        );
        let err = serde_json::from_value::<EmbeddingInput>(json!(42)).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid type: integer `42`, expected a string or an object with `parts`"
        );
        let err = serde_json::from_value::<EmbeddingInput>(json!({"part": []})).unwrap_err();
        assert_eq!(
            err.to_string(),
            "unknown field `part`, expected `parts` or `provider_data`"
        );
        let err = serde_json::from_str::<EmbeddingInput>(
            r#"{"parts": [{"type": "text", "text": "x"}], "provider-data": {}}"#,
        )
        .unwrap_err();
        assert!(
            err.to_string().starts_with("unknown field `provider-data`"),
            "{err}"
        );
    }

    #[test]
    fn with_provider_entry_inserts_a_typed_entry() {
        #[derive(Debug, PartialEq, Serialize, Deserialize)]
        struct Title(String);
        impl ProviderDataEntry for Title {
            const NAMESPACE: &'static str = "demo";
            const KEY: &'static str = "title";
        }
        let input = EmbeddingInput::text("x")
            .with_provider_entry(&Title("t".into()))
            .unwrap();
        assert_eq!(
            input.provider_data.get::<Title>().unwrap(),
            Some(Title("t".into()))
        );
    }

    #[test]
    fn image_parts_use_the_content_part_shape() {
        let part = EmbeddingPart::image_url("https://example.com/a.png");
        let value = serde_json::to_value(&part).unwrap();
        assert_eq!(
            value,
            json!({"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}})
        );
        assert_eq!(
            serde_json::to_value(ContentPart::image_url("https://example.com/a.png")).unwrap(),
            value
        );
        assert_eq!(
            serde_json::from_value::<EmbeddingPart>(value).unwrap(),
            part
        );
        assert_eq!(part.kind(), "image");
        assert!(part.as_text().is_none());
        assert!(part.as_image().is_some());
    }

    #[test]
    fn composite_inputs_round_trip() {
        let input = EmbeddingInput::from(vec![
            EmbeddingPart::text("A red shoe"),
            EmbeddingPart::image_base64("image/png", "AAAA"),
        ]);
        let value = serde_json::to_value(&input).unwrap();
        assert_eq!(
            value,
            json!({"parts": [
                {"type": "text", "text": "A red shoe"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}
            ]})
        );
        assert_eq!(
            serde_json::from_value::<EmbeddingInput>(value).unwrap(),
            input
        );
        assert_eq!(input.as_text(), None);
        assert_eq!(
            EmbeddingInput::from(crate::Image::url("https://e.x/a.png")),
            EmbeddingInput::image_url("https://e.x/a.png")
        );
    }

    #[test]
    fn inputs_work_as_cache_keys() {
        let mut set = std::collections::HashSet::new();
        set.insert(EmbeddingInput::from("a"));
        assert!(set.contains(&EmbeddingInput::text("a")));
        assert!(!set.contains(&EmbeddingInput::text("b")));
    }
}
