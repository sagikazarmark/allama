//! Media content shared by chat and embedding inputs: images, files, and
//! where their bytes come from.

use std::fmt;
use std::ops::Deref;
use std::sync::Arc;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::ProviderData;

/// Source of media content.
///
/// Build sources with the constructors ([`url`](Self::url),
/// [`base64`](Self::base64), [`bytes`](Self::bytes),
/// [`provider_file`](Self::provider_file)) and match them with `..`: the
/// enum and its `Base64` and `ProviderFile` variants are
/// `#[non_exhaustive]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum MediaSource {
    /// A remote resource.
    Url {
        /// Absolute URI. Each adapter accepts the schemes its provider
        /// documents (https almost everywhere; object-store schemes such as
        /// `gs://` where a provider reads them) and rejects others with
        /// [`Error::Unsupported`](crate::Error::Unsupported). Adapters never
        /// download a URL themselves.
        url: String,
    },
    /// Inline base64-encoded data.
    #[non_exhaustive]
    Base64 {
        /// MIME type of the decoded bytes.
        media_type: String,
        /// Base64-encoded bytes, shared rather than copied when the source
        /// is cloned (see [`Base64Data`]).
        data: Base64Data,
    },
    /// A file the provider stores, referenced by the id it issued (an
    /// OpenAI or Anthropic Files API `file_id`, a Mistral file id).
    ///
    /// The id means something only to the provider that issued it, so the
    /// source names that provider: `origin` is its provider-data namespace
    /// (`"openai"`, `"anthropic"`, …), as for
    /// [`OtherContent::origin`](crate::OtherContent::origin). A provider
    /// that does not [read](crate::ProviderIdentity::reads_namespace) the
    /// origin refuses the request with
    /// [`Error::Unsupported`](crate::Error::Unsupported) before sending
    /// (ADR 0002): sending an id it cannot resolve would fail at best and
    /// reach another account's file at worst. A provider file URI that is
    /// a URL (a Gemini Files API URI) is a [`Url`](Self::Url) source.
    #[non_exhaustive]
    ProviderFile {
        /// The namespace of the provider that issued the id.
        origin: String,
        /// The provider's id for the file.
        id: String,
    },
}

impl MediaSource {
    /// A URL source.
    #[must_use]
    pub fn url(url: impl Into<String>) -> Self {
        Self::Url { url: url.into() }
    }

    /// An inline base64 source.
    #[must_use]
    pub fn base64(media_type: impl Into<String>, data: impl Into<Base64Data>) -> Self {
        Self::Base64 {
            media_type: media_type.into(),
            data: data.into(),
        }
    }

    /// An inline source from raw bytes, such as a file read from disk,
    /// base64-encoded into a [`Base64`](Self::Base64) source.
    ///
    /// ```
    /// use allama::MediaSource;
    ///
    /// let source = MediaSource::bytes("image/png", [0x89, b'P', b'N', b'G']);
    /// assert_eq!(source, MediaSource::base64("image/png", "iVBORw=="));
    /// ```
    #[must_use]
    pub fn bytes(media_type: impl Into<String>, data: impl AsRef<[u8]>) -> Self {
        Self::base64(media_type, encode_base64(data.as_ref()))
    }

    /// A file stored by the provider whose namespace is `origin`, by the id
    /// it issued (see [`ProviderFile`](Self::ProviderFile)).
    ///
    /// ```
    /// use allama::MediaSource;
    ///
    /// let source = MediaSource::provider_file("openai", "file-abc123");
    /// assert_eq!(source.provider_file_id(), Some(("openai", "file-abc123")));
    /// ```
    #[must_use]
    pub fn provider_file(origin: impl Into<String>, id: impl Into<String>) -> Self {
        Self::ProviderFile {
            origin: origin.into(),
            id: id.into(),
        }
    }

    /// The media type of inline data.
    #[must_use]
    pub fn media_type(&self) -> Option<&str> {
        match self {
            Self::Base64 { media_type, .. } => Some(media_type),
            Self::Url { .. } | Self::ProviderFile { .. } => None,
        }
    }

    /// The base64 text of inline data.
    #[must_use]
    pub fn data(&self) -> Option<&str> {
        match self {
            Self::Base64 { data, .. } => Some(data.as_str()),
            Self::Url { .. } | Self::ProviderFile { .. } => None,
        }
    }

    /// The origin and id of a [provider file](Self::ProviderFile).
    #[must_use]
    pub fn provider_file_id(&self) -> Option<(&str, &str)> {
        match self {
            Self::ProviderFile { origin, id } => Some((origin, id)),
            Self::Url { .. } | Self::Base64 { .. } => None,
        }
    }
}

/// The base64 text of an inline [`MediaSource`]: cheap to clone.
///
/// Inline media is often the largest part of a request (a PDF, a few
/// images), and requests are cloned: by wrappers that rewrite them, by
/// fallback, by recording. The text is held behind a shared pointer, so a
/// clone shares it instead of copying it. It reads as a `&str`
/// ([`as_str`](Self::as_str), `Deref`, `Display`) and serializes as a JSON
/// string.
///
/// ```
/// use allama::{Base64Data, MediaSource};
///
/// let data = Base64Data::from("iVBORw==");
/// let source = MediaSource::base64("image/png", data.clone());
/// assert_eq!(source.data(), Some("iVBORw=="));
/// assert_eq!(&*data, "iVBORw==");
/// ```
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct Base64Data(Arc<str>);

impl Base64Data {
    /// The base64 text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Deref for Base64Data {
    type Target = str;

    fn deref(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for Base64Data {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for Base64Data {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&*self.0, f)
    }
}

impl fmt::Display for Base64Data {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for Base64Data {
    fn from(value: String) -> Self {
        Self(value.into())
    }
}

impl From<&str> for Base64Data {
    fn from(value: &str) -> Self {
        Self(value.into())
    }
}

impl From<Arc<str>> for Base64Data {
    fn from(value: Arc<str>) -> Self {
        Self(value)
    }
}

impl From<Base64Data> for String {
    fn from(value: Base64Data) -> Self {
        value.0.as_ref().to_owned()
    }
}

impl PartialEq<str> for Base64Data {
    fn eq(&self, other: &str) -> bool {
        &*self.0 == other
    }
}

impl PartialEq<&str> for Base64Data {
    fn eq(&self, other: &&str) -> bool {
        &*self.0 == *other
    }
}

impl Serialize for Base64Data {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for Base64Data {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        String::deserialize(deserializer).map(Self::from)
    }
}

/// Requested image fidelity for image input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ImageDetail {
    /// Let the provider decide.
    Auto,
    /// Low fidelity (cheaper).
    Low,
    /// High fidelity.
    High,
}

impl ImageDetail {
    /// Stable snake_case name.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Low => "low",
            Self::High => "high",
        }
    }
}

/// Image content (input or output).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Image {
    /// Image data or location.
    pub source: MediaSource,
    /// IANA media type of a [`MediaSource::Url`] image, for URLs whose path
    /// does not reveal it (object-store keys, provider file URIs). Inline
    /// data carries its own media type; when both are set they must agree.
    /// Adapters that need a media type fall back to the URL's extension.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    /// Requested fidelity for image input.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<ImageDetail>,
    /// Provider-specific data.
    #[serde(default, skip_serializing_if = "ProviderData::is_empty")]
    pub provider_data: ProviderData,
}

impl Image {
    /// Create image content.
    #[must_use]
    pub fn new(source: MediaSource) -> Self {
        Self {
            source,
            media_type: None,
            detail: None,
            provider_data: ProviderData::new(),
        }
    }

    /// Image from a URL.
    #[must_use]
    pub fn url(url: impl Into<String>) -> Self {
        Self::new(MediaSource::url(url))
    }

    /// Image from inline base64 data.
    #[must_use]
    pub fn base64(media_type: impl Into<String>, data: impl Into<Base64Data>) -> Self {
        Self::new(MediaSource::base64(media_type, data))
    }

    /// Image from raw bytes, sent inline; see [`MediaSource::bytes`].
    #[must_use]
    pub fn bytes(media_type: impl Into<String>, data: impl AsRef<[u8]>) -> Self {
        Self::new(MediaSource::bytes(media_type, data))
    }

    /// Image stored by a provider; see [`MediaSource::provider_file`].
    #[must_use]
    pub fn provider_file(origin: impl Into<String>, id: impl Into<String>) -> Self {
        Self::new(MediaSource::provider_file(origin, id))
    }

    /// Set the media type of a URL image.
    #[must_use]
    pub fn with_media_type(mut self, media_type: impl Into<String>) -> Self {
        self.media_type = Some(media_type.into());
        self
    }

    /// The image's media type: the inline data's, else [`media_type`](Self::media_type).
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidRequest`](crate::Error::InvalidRequest) when
    /// inline data and [`media_type`](Self::media_type) disagree.
    pub fn resolved_media_type(&self) -> crate::Result<Option<&str>> {
        match (&self.source, self.media_type.as_deref()) {
            (MediaSource::Base64 { media_type, .. }, Some(declared))
                if !media_type.eq_ignore_ascii_case(declared) =>
            {
                Err(crate::Error::invalid_request(format!(
                    "image media type `{declared}` disagrees with its inline data (`{media_type}`)"
                ))
                .authored())
            }
            (MediaSource::Base64 { media_type, .. }, _) => Ok(Some(media_type)),
            (_, declared) => Ok(declared),
        }
    }

    /// Set the requested fidelity.
    #[must_use]
    pub fn with_detail(mut self, detail: ImageDetail) -> Self {
        self.detail = Some(detail);
        self
    }

    /// Attach provider data.
    #[must_use]
    pub fn with_provider_data(mut self, provider_data: ProviderData) -> Self {
        self.provider_data = provider_data;
        self
    }
}

/// File input: a document (PDF, plain text, …), audio, or video.
///
/// What a provider takes depends on the media type:
/// [`ChatCapability::FileInput`](crate::ChatCapability::FileInput) covers
/// documents, [`AudioInput`](crate::ChatCapability::AudioInput) an
/// `audio/*` file and [`VideoInput`](crate::ChatCapability::VideoInput) a
/// `video/*` one.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub struct File {
    /// IANA media type, e.g. `application/pdf`.
    pub media_type: String,
    /// File data or location.
    pub source: MediaSource,
    /// Optional file name shown to the model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    /// Provider-specific data.
    #[serde(default, skip_serializing_if = "ProviderData::is_empty")]
    pub provider_data: ProviderData,
}

impl File {
    /// File from inline base64 data.
    #[must_use]
    pub fn base64(media_type: impl Into<String>, data: impl Into<Base64Data>) -> Self {
        let media_type = media_type.into();
        Self::new(media_type.clone(), MediaSource::base64(media_type, data))
    }

    /// File from raw bytes, sent inline; see [`MediaSource::bytes`].
    #[must_use]
    pub fn bytes(media_type: impl Into<String>, data: impl AsRef<[u8]>) -> Self {
        Self::base64(media_type, encode_base64(data.as_ref()))
    }

    /// File from a URL.
    #[must_use]
    pub fn url(media_type: impl Into<String>, url: impl Into<String>) -> Self {
        Self::new(media_type, MediaSource::url(url))
    }

    /// File stored by a provider; see [`MediaSource::provider_file`].
    ///
    /// ```
    /// use allama::File;
    ///
    /// let file = File::provider_file("application/pdf", "anthropic", "file_011CNha8iCJcU1wXNR6q4V8w");
    /// assert_eq!(file.source.provider_file_id().map(|(origin, _)| origin), Some("anthropic"));
    /// ```
    #[must_use]
    pub fn provider_file(
        media_type: impl Into<String>,
        origin: impl Into<String>,
        id: impl Into<String>,
    ) -> Self {
        Self::new(media_type, MediaSource::provider_file(origin, id))
    }

    /// File from any source.
    #[must_use]
    pub fn new(media_type: impl Into<String>, source: MediaSource) -> Self {
        Self {
            media_type: media_type.into(),
            source,
            filename: None,
            provider_data: ProviderData::new(),
        }
    }

    /// The kind of media the file holds, from its media type.
    #[must_use]
    pub fn kind(&self) -> FileKind {
        FileKind::of(&self.media_type)
    }

    /// Set the file name.
    #[must_use]
    pub fn with_filename(mut self, filename: impl Into<String>) -> Self {
        self.filename = Some(filename.into());
        self
    }

    /// Attach provider data.
    #[must_use]
    pub fn with_provider_data(mut self, provider_data: ProviderData) -> Self {
        self.provider_data = provider_data;
        self
    }
}

/// The kind of media a [`File`] holds, from its media type: what decides
/// the capability a provider needs to take it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FileKind {
    /// A document: any media type that is not audio or video (PDF, plain
    /// text, …). Needs [`ChatCapability::FileInput`](crate::ChatCapability::FileInput).
    Document,
    /// An `audio/*` file. Needs [`ChatCapability::AudioInput`](crate::ChatCapability::AudioInput).
    Audio,
    /// A `video/*` file. Needs [`ChatCapability::VideoInput`](crate::ChatCapability::VideoInput).
    Video,
}

impl FileKind {
    /// The kind of a media type: `audio/*` is audio, `video/*` video, and
    /// anything else a document. Parameters and case are ignored.
    ///
    /// ```
    /// use allama::FileKind;
    ///
    /// assert_eq!(FileKind::of("audio/wav"), FileKind::Audio);
    /// assert_eq!(FileKind::of("Video/MP4"), FileKind::Video);
    /// assert_eq!(FileKind::of("application/pdf"), FileKind::Document);
    /// ```
    #[must_use]
    pub fn of(media_type: &str) -> Self {
        let top = media_type.split('/').next().unwrap_or_default().trim();
        if top.eq_ignore_ascii_case("audio") {
            Self::Audio
        } else if top.eq_ignore_ascii_case("video") {
            Self::Video
        } else {
            Self::Document
        }
    }
}

/// Standard base64 (RFC 4648 §4), with padding: the encoding of every
/// [`MediaSource::Base64`] source.
fn encode_base64(data: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut encoded = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let bytes = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let group = u32::from_be_bytes([0, bytes[0], bytes[1], bytes[2]]);
        for i in 0..4 {
            if i <= chunk.len() {
                let index = (group >> (18 - 6 * i)) & 0x3f;
                encoded.push(char::from(ALPHABET[index as usize]));
            } else {
                encoded.push('=');
            }
        }
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_encoding_follows_rfc_4648() {
        // The test vectors of RFC 4648 §10.
        for (input, expected) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
        ] {
            assert_eq!(encode_base64(input.as_bytes()), expected, "{input:?}");
        }
        // Every byte value, including the last two letters of the alphabet.
        assert_eq!(encode_base64(&[0xfb, 0xff, 0xbf]), "+/+/");
        assert_eq!(encode_base64(&[0, 0x10, 0x83]), "ABCD");
    }

    #[test]
    fn bytes_constructors_send_the_bytes_inline() {
        let png = [0x89, b'P', b'N', b'G'];
        assert_eq!(
            Image::bytes("image/png", png),
            Image::base64("image/png", "iVBORw==")
        );
        assert_eq!(
            File::bytes("application/pdf", b"%PDF-").with_filename("a.pdf"),
            File::base64("application/pdf", "JVBERi0=").with_filename("a.pdf")
        );
    }

    #[test]
    fn resolved_media_type_prefers_inline_data_and_rejects_conflicts() {
        assert_eq!(
            Image::base64("image/png", "AA")
                .resolved_media_type()
                .unwrap(),
            Some("image/png")
        );
        assert_eq!(
            Image::base64("image/png", "AA")
                .with_media_type("IMAGE/PNG")
                .resolved_media_type()
                .unwrap(),
            Some("image/png")
        );
        assert!(
            Image::base64("image/png", "AA")
                .with_media_type("image/jpeg")
                .resolved_media_type()
                .is_err()
        );
        assert_eq!(
            Image::url("s3://bucket/key")
                .with_media_type("image/webp")
                .resolved_media_type()
                .unwrap(),
            Some("image/webp")
        );
        assert_eq!(
            Image::url("https://e.x/a.png")
                .resolved_media_type()
                .unwrap(),
            None
        );
    }

    /// Cloning a request shares its inline media instead of copying it.
    #[test]
    fn cloned_inline_data_is_shared() {
        let request = crate::ChatRequest::new("m").message(crate::Message::user_parts(vec![
            crate::ContentPart::image_base64("image/png", "A".repeat(1 << 16)),
        ]));
        let data = |request: &crate::ChatRequest| {
            let crate::Message::User(user) = &request.messages[0] else {
                unreachable!()
            };
            let crate::UserContent::Parts(parts) = &user.content else {
                unreachable!()
            };
            let crate::ContentPart::Image(image) = &parts[0] else {
                unreachable!()
            };
            image.source.data().unwrap().as_ptr()
        };
        assert_eq!(data(&request), data(&request.clone()));
    }

    #[test]
    fn sources_serialize_with_their_type_tag() {
        let base64 = MediaSource::base64("image/png", "AA");
        assert_eq!(
            serde_json::to_value(&base64).unwrap(),
            serde_json::json!({"type": "base64", "media_type": "image/png", "data": "AA"})
        );
        let file = MediaSource::provider_file("openai", "file-1");
        let value = serde_json::to_value(&file).unwrap();
        assert_eq!(
            value,
            serde_json::json!({"type": "provider_file", "origin": "openai", "id": "file-1"})
        );
        assert_eq!(serde_json::from_value::<MediaSource>(value).unwrap(), file);
        assert_eq!(file.data(), None);
        assert_eq!(base64.data(), Some("AA"));
    }

    #[test]
    fn file_kind_follows_the_media_type() {
        assert_eq!(File::base64("audio/mpeg", "AA").kind(), FileKind::Audio);
        assert_eq!(
            File::url("video/webm", "https://e.x/v").kind(),
            FileKind::Video
        );
        assert_eq!(
            File::base64("application/pdf", "AA").kind(),
            FileKind::Document
        );
        assert_eq!(
            FileKind::of("text/plain; charset=utf-8"),
            FileKind::Document
        );
    }

    #[test]
    fn media_type_is_omitted_when_unset() {
        let value = serde_json::to_value(Image::url("https://e.x/a.png")).unwrap();
        assert_eq!(
            value,
            serde_json::json!({"source": {"type": "url", "url": "https://e.x/a.png"}})
        );
    }
}
