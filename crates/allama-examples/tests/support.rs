use allama::{DynChatProvider, DynEmbeddingProvider, Error, Result};

/// A provider the live tests run against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KnownProvider {
    /// Provider key in `allama_providers` (`openai`, `anthropic`, ...).
    pub key: &'static str,
    /// The credential its `from_env` constructor requires.
    pub credential_env: &'static str,
    /// Default chat model id.
    pub chat_model: &'static str,
    /// Default embedding model id, for providers that serve embeddings.
    pub embedding_model: Option<&'static str>,
}

/// The providers the live tests run, each at its default model: the one
/// maintained list of default model ids, which the crate README mirrors.
///
/// `openai` is the OpenAI Responses API and `openai-chat` is Chat
/// Completions; both use `OPENAI_API_KEY`, and `openai` is the default.
pub const KNOWN_PROVIDERS: [KnownProvider; 4] = [
    KnownProvider {
        key: "anthropic",
        credential_env: "ANTHROPIC_API_KEY",
        chat_model: "claude-haiku-4-5",
        embedding_model: None,
    },
    KnownProvider {
        key: "gemini",
        credential_env: "GEMINI_API_KEY",
        chat_model: "gemini-3.5-flash",
        embedding_model: Some("gemini-embedding-001"),
    },
    KnownProvider {
        key: "openai",
        credential_env: "OPENAI_API_KEY",
        chat_model: "gpt-5-mini",
        embedding_model: Some("text-embedding-3-small"),
    },
    KnownProvider {
        key: "openai-chat",
        credential_env: "OPENAI_API_KEY",
        chat_model: "gpt-5-mini",
        embedding_model: None,
    },
];

/// A provider built from a model address, with the model to send.
pub struct Loaded<P> {
    /// The provider.
    pub provider: P,
    /// The model id to put on requests.
    pub model: String,
    /// The model address the provider was built from.
    pub address: String,
}

impl KnownProvider {
    /// Look up a known provider by key.
    #[must_use]
    pub fn get(key: &str) -> Option<Self> {
        KNOWN_PROVIDERS.into_iter().find(|known| known.key == key)
    }

    /// The default chat model address, such as `openai:gpt-5-mini`.
    #[must_use]
    pub fn chat_address(&self) -> String {
        format!("{}:{}", self.key, self.chat_model)
    }

    /// The default embedding model address, when the provider has one.
    #[must_use]
    pub fn embedding_address(&self) -> Option<String> {
        self.embedding_model
            .map(|model| format!("{}:{model}", self.key))
    }

    /// Whether the provider's credential is set.
    #[must_use]
    pub fn is_configured(&self) -> bool {
        env_value(self.credential_env).is_some()
    }

    /// Build the chat provider for the default chat model.
    ///
    /// # Errors
    ///
    /// The provider's own errors, such as [`Error::Configuration`] for a
    /// missing credential.
    pub fn load_chat(&self) -> Result<Loaded<DynChatProvider>> {
        load_chat_address(self.chat_address())
    }

    /// Build the embedding provider for the default embedding model, when
    /// the provider has one.
    ///
    /// # Errors
    ///
    /// The provider's own errors, such as [`Error::Configuration`] for a
    /// missing credential.
    pub fn load_embedding(&self) -> Result<Option<Loaded<DynEmbeddingProvider>>> {
        self.embedding_address()
            .map(load_embedding_address)
            .transpose()
    }
}

/// Selection aliases for every configured provider.
const ALIASES: &[&str] = &["all", "configured"];

pub fn live_selection() -> Option<String> {
    std::env::var("ALLAMA_LIVE_PROVIDER")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

pub fn validate_live_selection(selection: &str) -> Result<()> {
    for raw in selection.split(',') {
        let name = raw.trim();
        if name.is_empty() {
            return Err(Error::invalid_request(
                "ALLAMA_LIVE_PROVIDER contains an empty provider name",
            ));
        }

        if !ALIASES.contains(&name) && KnownProvider::get(name).is_none() {
            let valid: Vec<&str> = ALIASES
                .iter()
                .copied()
                .chain(KNOWN_PROVIDERS.iter().map(|known| known.key))
                .collect();
            return Err(Error::invalid_request(format!(
                "unsupported ALLAMA_LIVE_PROVIDER entry `{name}`; expected one of: {}",
                valid.join(", ")
            )));
        }
    }

    Ok(())
}

pub fn selection_includes(selection: &str, name: &str) -> bool {
    selection
        .split(',')
        .map(str::trim)
        .any(|value| ALIASES.contains(&value) || value == name)
}

pub fn selection_uses_configured_alias(selection: &str) -> bool {
    selection
        .split(',')
        .map(str::trim)
        .any(|value| ALIASES.contains(&value))
}

pub fn ensure_selection_has_runnable_target(selection: &str) -> Result<()> {
    let has_runnable_target = KNOWN_PROVIDERS
        .iter()
        .any(|known| selection_includes(selection, known.key) && known.is_configured());

    if has_runnable_target {
        Ok(())
    } else {
        Err(Error::invalid_request(format!(
            "ALLAMA_LIVE_PROVIDER={selection} selected no runnable live targets; configure at least one provider credential"
        )))
    }
}

pub fn assert_contains_token(text: &str, token: &str) {
    assert!(
        text.to_ascii_lowercase()
            .contains(&token.to_ascii_lowercase()),
        "expected response to contain '{token}', got: {text}"
    );
}

fn load_chat_address(address: String) -> Result<Loaded<DynChatProvider>> {
    let (provider, model) = allama_providers::chat::from_model(&address)?;
    Ok(Loaded {
        provider,
        model,
        address,
    })
}

fn load_embedding_address(address: String) -> Result<Loaded<DynEmbeddingProvider>> {
    let (provider, model) = allama_providers::embedding::from_model(&address)?;
    Ok(Loaded {
        provider,
        model,
        address,
    })
}

fn env_value(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unknown_live_selection_entry() {
        let err = validate_live_selection("openai,opneai").unwrap_err();
        assert!(matches!(err, Error::InvalidRequest { message, .. } if message.contains("opneai")));
    }

    #[test]
    fn rejects_empty_live_selection_entry() {
        let err = validate_live_selection("openai,").unwrap_err();
        assert!(
            matches!(err, Error::InvalidRequest { message, .. } if message.contains("empty provider name"))
        );
    }

    #[test]
    fn accepts_known_live_selection_entries() {
        validate_live_selection("configured,openai,openai-chat").unwrap();
    }
}
