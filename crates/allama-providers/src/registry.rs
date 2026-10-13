use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, LazyLock};

use allama::{
    ChatProvider, DecisionProvider, DynChatProvider, DynDecisionProvider, DynEmbeddingProvider,
    EmbeddingProvider, Error, Result,
};

use crate::address::is_valid_key;
use crate::{AdapterSettings, ModelAddress, ProviderSettings, builtin, stack};

/// Provider constructors by provider key.
///
/// Each key maps to at most one constructor per operation. A constructor
/// runs on every lookup, so each call returns a fresh provider; keep the
/// result rather than looking it up per request.
///
/// The free functions in [`chat`](crate::chat),
/// [`embedding`](crate::embedding), and [`decision`](crate::decision) use
/// [`Registry::builtin`].
///
/// # Settings
///
/// The `_with` lookups take [`ProviderSettings`] from the caller, such as a
/// credential scope; the others use the defaults. Every lookup wraps what
/// the constructor built in the settings' stack: retries (on by default),
/// a per-attempt timeout and an unsupported-feature policy (see
/// [The default stack](ProviderSettings#the-default-stack)), whichever way
/// the constructor was registered.
///
/// A constructor registered with `register_chat_with` (and its embedding
/// and decision twins) also builds the provider with the settings only an
/// adapter can apply, which it takes from [`AdapterSettings`]: the built-in
/// providers apply the credential scope with their `with_credential_scope`,
/// and chat capability overrides with their `with_chat_capabilities`. A
/// lookup that sets one of them fails for its key with
/// [`Error::Configuration`] when the constructor leaves it untaken, or was
/// registered without settings, instead of building a provider that would
/// send the wrong account's key or answer for the wrong model.
///
/// ```no_run
/// # #[cfg(feature = "openai")] {
/// use allama_providers::{ProviderSettings, Registry};
///
/// # fn example() -> allama::Result<()> {
/// let registry = Registry::builtin();
/// let settings = ProviderSettings::new().credential_scope("tenant-openai");
/// let (provider, model) = registry.chat_from_model_with("openai:gpt-5.5", &settings)?;
/// # let _ = (provider, model); Ok(()) }
/// # }
/// ```
#[derive(Clone, Default)]
pub struct Registry {
    providers: BTreeMap<String, Constructors>,
}

impl Registry {
    /// An empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A registry with every built-in provider enabled by crate features,
    /// each configured from the environment.
    #[must_use]
    pub fn builtin() -> Self {
        let mut registry = Self::new();
        builtin::register(&mut registry);
        registry
    }

    /// Register the chat provider for `key`, replacing any previous one.
    ///
    /// The constructor takes no [`ProviderSettings`], so a lookup with a
    /// credential scope or chat capability overrides fails for `key` (see
    /// [Settings](Registry#settings)); use
    /// [`register_chat_with`](Self::register_chat_with) to honour them. The
    /// lookup's stack (retries included) wraps the provider either way.
    ///
    /// # Panics
    ///
    /// Panics when `key` is not a valid provider key (lowercase ASCII
    /// letters, digits and `-`, starting with a letter or digit; see
    /// [`ModelAddress`]), since an address could not name it.
    pub fn register_chat<F, P>(&mut self, key: impl Into<String>, constructor: F) -> &mut Self
    where
        F: Fn() -> Result<P> + Send + Sync + 'static,
        P: ChatProvider + 'static,
    {
        let key = key.into();
        let constructor = without_settings(key.clone(), constructor);
        self.register_chat_with(key, constructor)
    }

    /// Register the chat provider for `key`, built with the lookup's
    /// [`AdapterSettings`], replacing any previous one.
    ///
    /// The constructor takes each setting it applies to the adapter. A
    /// lookup that sets a credential scope or chat capability overrides the
    /// constructor did not take fails for `key` (see
    /// [Settings](Registry#settings)). The lookup's stack (retries
    /// included) wraps the provider.
    ///
    /// # Panics
    ///
    /// Panics when `key` is not a valid provider key (lowercase ASCII
    /// letters, digits and `-`, starting with a letter or digit; see
    /// [`ModelAddress`]), since an address could not name it.
    pub fn register_chat_with<F, P>(&mut self, key: impl Into<String>, constructor: F) -> &mut Self
    where
        F: Fn(&mut AdapterSettings) -> Result<P> + Send + Sync + 'static,
        P: ChatProvider + 'static,
    {
        let key = key.into();
        let entry = self.entry(key.clone());
        entry.chat = Some(Arc::new(move |settings| {
            let provider = build(&key, Operation::Chat, settings, &constructor)?;
            Ok(stack::chat(provider, settings))
        }));
        self
    }

    /// Register the embedding provider for `key`, replacing any previous
    /// one. Like [`register_chat`](Self::register_chat), it takes no
    /// [`ProviderSettings`].
    ///
    /// # Panics
    ///
    /// Panics when `key` is not a valid provider key (lowercase ASCII
    /// letters, digits and `-`, starting with a letter or digit; see
    /// [`ModelAddress`]), since an address could not name it.
    pub fn register_embedding<F, P>(&mut self, key: impl Into<String>, constructor: F) -> &mut Self
    where
        F: Fn() -> Result<P> + Send + Sync + 'static,
        P: EmbeddingProvider + 'static,
    {
        let key = key.into();
        let constructor = without_settings(key.clone(), constructor);
        self.register_embedding_with(key, constructor)
    }

    /// Register the embedding provider for `key`, built with the lookup's
    /// [`AdapterSettings`], replacing any previous one. Like
    /// [`register_chat_with`](Self::register_chat_with), a lookup fails when
    /// the constructor leaves a setting untaken.
    ///
    /// # Panics
    ///
    /// Panics when `key` is not a valid provider key (lowercase ASCII
    /// letters, digits and `-`, starting with a letter or digit; see
    /// [`ModelAddress`]), since an address could not name it.
    pub fn register_embedding_with<F, P>(
        &mut self,
        key: impl Into<String>,
        constructor: F,
    ) -> &mut Self
    where
        F: Fn(&mut AdapterSettings) -> Result<P> + Send + Sync + 'static,
        P: EmbeddingProvider + 'static,
    {
        let key = key.into();
        let entry = self.entry(key.clone());
        entry.embedding = Some(Arc::new(move |settings| {
            let provider = build(&key, Operation::Embedding, settings, &constructor)?;
            Ok(stack::embedding(provider, settings))
        }));
        self
    }

    /// Register the decision provider for `key`, replacing any previous
    /// one. Like [`register_chat`](Self::register_chat), it takes no
    /// [`ProviderSettings`].
    ///
    /// # Panics
    ///
    /// Panics when `key` is not a valid provider key (lowercase ASCII
    /// letters, digits and `-`, starting with a letter or digit; see
    /// [`ModelAddress`]), since an address could not name it.
    pub fn register_decision<F, P>(&mut self, key: impl Into<String>, constructor: F) -> &mut Self
    where
        F: Fn() -> Result<P> + Send + Sync + 'static,
        P: DecisionProvider + 'static,
    {
        let key = key.into();
        let constructor = without_settings(key.clone(), constructor);
        self.register_decision_with(key, constructor)
    }

    /// Register the decision provider for `key`, built with the lookup's
    /// [`AdapterSettings`], replacing any previous one. Like
    /// [`register_chat_with`](Self::register_chat_with), a lookup fails when
    /// the constructor leaves a setting untaken.
    ///
    /// # Panics
    ///
    /// Panics when `key` is not a valid provider key (lowercase ASCII
    /// letters, digits and `-`, starting with a letter or digit; see
    /// [`ModelAddress`]), since an address could not name it.
    pub fn register_decision_with<F, P>(
        &mut self,
        key: impl Into<String>,
        constructor: F,
    ) -> &mut Self
    where
        F: Fn(&mut AdapterSettings) -> Result<P> + Send + Sync + 'static,
        P: DecisionProvider + 'static,
    {
        let key = key.into();
        let entry = self.entry(key.clone());
        entry.decision = Some(Arc::new(move |settings| {
            let provider = build(&key, Operation::Decision, settings, &constructor)?;
            Ok(stack::decision(provider, settings))
        }));
        self
    }

    /// The registered provider keys, sorted.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.providers.keys().map(String::as_str)
    }

    /// Build the chat provider registered for `key`.
    ///
    /// # Errors
    ///
    /// Fails with [`Error::Configuration`] when `key` is unknown or is
    /// registered without a chat provider (the message lists the keys that
    /// offer chat), and with the constructor's error otherwise (for example
    /// [`Error::Configuration`] for a missing API key).
    pub fn chat_from_provider(&self, key: &str) -> Result<DynChatProvider> {
        self.chat_from_provider_with(key, &ProviderSettings::default())
    }

    /// Build the chat provider registered for `key` with `settings`, such
    /// as a credential scope.
    ///
    /// # Errors
    ///
    /// As [`Registry::chat_from_provider`].
    pub fn chat_from_provider_with(
        &self,
        key: &str,
        settings: &ProviderSettings,
    ) -> Result<DynChatProvider> {
        match self.providers.get(key).and_then(|c| c.chat.as_ref()) {
            Some(constructor) => constructor(settings),
            None => Err(self.missing(key, Operation::Chat)),
        }
    }

    /// Build the chat provider for a `provider:model` address and return it
    /// with the model id.
    ///
    /// # Errors
    ///
    /// Fails with [`Error::Configuration`] when the address is not a
    /// [`ModelAddress`], and as [`Registry::chat_from_provider`] otherwise.
    pub fn chat_from_model(&self, address: &str) -> Result<(DynChatProvider, String)> {
        self.chat_from_model_with(address, &ProviderSettings::default())
    }

    /// Build the chat provider for a `provider:model` address with
    /// `settings`, and return it with the model id.
    ///
    /// # Errors
    ///
    /// As [`Registry::chat_from_model`].
    pub fn chat_from_model_with(
        &self,
        address: &str,
        settings: &ProviderSettings,
    ) -> Result<(DynChatProvider, String)> {
        let (key, model) = self.parse_address(address)?.into_parts();
        Ok((self.chat_from_provider_with(&key, settings)?, model))
    }

    /// Build the embedding provider registered for `key`.
    ///
    /// # Errors
    ///
    /// Fails with [`Error::Configuration`] when `key` is unknown or is
    /// registered without an embedding provider (the message lists the keys
    /// that offer embeddings), and with the constructor's error otherwise.
    pub fn embedding_from_provider(&self, key: &str) -> Result<DynEmbeddingProvider> {
        self.embedding_from_provider_with(key, &ProviderSettings::default())
    }

    /// Build the embedding provider registered for `key` with `settings`, such
    /// as a credential scope.
    ///
    /// # Errors
    ///
    /// As [`Registry::embedding_from_provider`].
    pub fn embedding_from_provider_with(
        &self,
        key: &str,
        settings: &ProviderSettings,
    ) -> Result<DynEmbeddingProvider> {
        match self.providers.get(key).and_then(|c| c.embedding.as_ref()) {
            Some(constructor) => constructor(settings),
            None => Err(self.missing(key, Operation::Embedding)),
        }
    }

    /// Build the embedding provider for a `provider:model` address and
    /// return it with the model id.
    ///
    /// # Errors
    ///
    /// Fails with [`Error::Configuration`] when the address is not a
    /// [`ModelAddress`], and as [`Registry::embedding_from_provider`]
    /// otherwise.
    pub fn embedding_from_model(&self, address: &str) -> Result<(DynEmbeddingProvider, String)> {
        self.embedding_from_model_with(address, &ProviderSettings::default())
    }

    /// Build the embedding provider for a `provider:model` address with
    /// `settings`, and return it with the model id.
    ///
    /// # Errors
    ///
    /// As [`Registry::embedding_from_model`].
    pub fn embedding_from_model_with(
        &self,
        address: &str,
        settings: &ProviderSettings,
    ) -> Result<(DynEmbeddingProvider, String)> {
        let (key, model) = self.parse_address(address)?.into_parts();
        Ok((self.embedding_from_provider_with(&key, settings)?, model))
    }

    /// Build the decision provider registered for `key`.
    ///
    /// # Errors
    ///
    /// Fails with [`Error::Configuration`] when `key` is unknown or is
    /// registered without a decision provider (the message lists the keys
    /// that offer decisions), and with the constructor's error otherwise.
    pub fn decision_from_provider(&self, key: &str) -> Result<DynDecisionProvider> {
        self.decision_from_provider_with(key, &ProviderSettings::default())
    }

    /// Build the decision provider registered for `key` with `settings`, such
    /// as a credential scope.
    ///
    /// # Errors
    ///
    /// As [`Registry::decision_from_provider`].
    pub fn decision_from_provider_with(
        &self,
        key: &str,
        settings: &ProviderSettings,
    ) -> Result<DynDecisionProvider> {
        match self.providers.get(key).and_then(|c| c.decision.as_ref()) {
            Some(constructor) => constructor(settings),
            None => Err(self.missing(key, Operation::Decision)),
        }
    }

    /// Build the decision provider for a `provider:model` address and
    /// return it with the model id.
    ///
    /// # Errors
    ///
    /// Fails with [`Error::Configuration`] when the address is not a
    /// [`ModelAddress`], and as [`Registry::decision_from_provider`]
    /// otherwise.
    pub fn decision_from_model(&self, address: &str) -> Result<(DynDecisionProvider, String)> {
        self.decision_from_model_with(address, &ProviderSettings::default())
    }

    /// Build the decision provider for a `provider:model` address with
    /// `settings`, and return it with the model id.
    ///
    /// # Errors
    ///
    /// As [`Registry::decision_from_model`].
    pub fn decision_from_model_with(
        &self,
        address: &str,
        settings: &ProviderSettings,
    ) -> Result<(DynDecisionProvider, String)> {
        let (key, model) = self.parse_address(address)?.into_parts();
        Ok((self.decision_from_provider_with(&key, settings)?, model))
    }

    /// Whether `key` has a constructor for `operation`.
    #[cfg(test)]
    pub(crate) fn supports(&self, key: &str, operation: Operation) -> bool {
        self.providers
            .get(key)
            .is_some_and(|constructors| constructors.offers(operation))
    }

    fn entry(&mut self, key: impl Into<String>) -> &mut Constructors {
        let key = key.into();
        assert!(
            is_valid_key(&key),
            "invalid provider key {key:?}: keys are lowercase ASCII letters, digits and `-`, \
             starting with a letter or digit"
        );
        self.providers.entry(key).or_default()
    }

    /// Parse a model address, with a hint when it looks like an address of
    /// a registered key written another way (`openai/gpt-5`,
    /// `OpenAI:gpt-5`).
    fn parse_address(&self, address: &str) -> Result<ModelAddress> {
        address
            .parse()
            .map_err(|err: Error| match self.suggest(address) {
                Some(suggestion) => {
                    Error::configuration(format!("{}; did you mean `{suggestion}`?", err.message()))
                        .authored()
                }
                None => err,
            })
    }

    /// The address of a registered key that `address` most likely means:
    /// trimmed, its key lowercased, or its first `/` read as the `:`.
    fn suggest(&self, address: &str) -> Option<String> {
        let trimmed = address.trim();
        let (key, model) = trimmed
            .split_once(':')
            .or_else(|| trimmed.split_once('/'))?;
        let key = key.trim().to_ascii_lowercase();
        let model = model.trim();
        let suggestion = format!("{key}:{model}");
        (self.providers.contains_key(&key) && suggestion.parse::<ModelAddress>().is_ok())
            .then_some(suggestion)
    }

    /// The keys that offer `operation`, sorted.
    fn keys_for(&self, operation: Operation) -> Vec<&str> {
        self.providers
            .iter()
            .filter(|(_, constructors)| constructors.offers(operation))
            .map(|(key, _)| key.as_str())
            .collect()
    }

    /// The configuration error for a lookup of `operation` under `key` that
    /// has no constructor: it names the feature that would add one, or the
    /// keys that offer the operation.
    fn missing(&self, key: &str, operation: Operation) -> Error {
        if let Some(feature) = builtin::disabled_feature(key, operation) {
            return Error::configuration(format!(
                "{operation} with provider `{key}` requires the `{feature}` feature of allama-providers"
            )).authored();
        }
        let noun = operation.provider_noun();
        let keys = self.keys_for(operation);
        let available = if keys.is_empty() {
            format!("no {noun} providers are registered")
        } else {
            format!("{noun} providers: {}", keys.join(", "))
        };
        let message = match self.providers.get(key) {
            Some(constructors) => {
                let offered = Operation::ALL
                    .into_iter()
                    .filter(|operation| constructors.offers(*operation))
                    .map(|operation| operation.to_string())
                    .collect::<Vec<_>>();
                format!(
                    "provider `{key}` does not offer {operation} (it offers {}); {available}",
                    offered.join(" and ")
                )
            }
            None => {
                let normalized = key.trim().to_ascii_lowercase();
                let hint = if normalized != key && self.providers.contains_key(&normalized) {
                    format!("did you mean `{normalized}`? ")
                } else {
                    String::new()
                };
                format!(
                    "unknown {noun} provider `{key}`; {hint}{available}; register your own with \
                     `Registry::register_{noun}`"
                )
            }
        };
        Error::configuration(message).authored()
    }
}

impl fmt::Debug for Registry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(&self.providers).finish()
    }
}

type ChatConstructor = Arc<dyn Fn(&ProviderSettings) -> Result<DynChatProvider> + Send + Sync>;
type EmbeddingConstructor =
    Arc<dyn Fn(&ProviderSettings) -> Result<DynEmbeddingProvider> + Send + Sync>;
type DecisionConstructor =
    Arc<dyn Fn(&ProviderSettings) -> Result<DynDecisionProvider> + Send + Sync>;

#[derive(Clone, Default)]
struct Constructors {
    chat: Option<ChatConstructor>,
    embedding: Option<EmbeddingConstructor>,
    decision: Option<DecisionConstructor>,
}

impl Constructors {
    fn offers(&self, operation: Operation) -> bool {
        match operation {
            Operation::Chat => self.chat.is_some(),
            Operation::Embedding => self.embedding.is_some(),
            Operation::Decision => self.decision.is_some(),
        }
    }
}

impl fmt::Debug for Constructors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Constructors")
            .field("chat", &self.chat.is_some())
            .field("embedding", &self.embedding.is_some())
            .field("decision", &self.decision.is_some())
            .finish()
    }
}

/// A model operation a provider key can offer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Operation {
    Chat,
    Embedding,
    Decision,
}

impl Operation {
    const ALL: [Self; 3] = [Self::Chat, Self::Embedding, Self::Decision];

    /// The operation as a modifier of "provider" ("chat provider",
    /// "embedding provider"), which is also its module's name.
    pub(crate) fn provider_noun(self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Embedding => "embedding",
            Self::Decision => "decision",
        }
    }
}

impl fmt::Display for Operation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Chat => "chat",
            Self::Embedding => "embeddings",
            Self::Decision => "decisions",
        })
    }
}

/// The shared [`Registry::builtin`] behind the free functions.
pub(crate) fn builtin() -> &'static Registry {
    static BUILTIN: LazyLock<Registry> = LazyLock::new(Registry::builtin);
    &BUILTIN
}

/// Run `constructor`, registered for `key`, with the settings of
/// `settings` it must apply for `operation`, and refuse what it built when
/// it left one of them untaken.
fn build<F, P>(
    key: &str,
    operation: Operation,
    settings: &ProviderSettings,
    constructor: &F,
) -> Result<P>
where
    F: Fn(&mut AdapterSettings) -> Result<P>,
{
    let mut adapter_settings = AdapterSettings::new(settings, operation);
    let provider = constructor(&mut adapter_settings)?;
    match adapter_settings.untaken() {
        Some(setting) => Err(setting.untaken_by(key)),
        None => Ok(provider),
    }
}

/// A settings-aware constructor for one that takes no settings: it builds
/// without them, and refuses settings only a constructor could apply,
/// before building. The lookup still wraps what it builds.
fn without_settings<F, P>(
    key: String,
    constructor: F,
) -> impl Fn(&mut AdapterSettings) -> Result<P> + Send + Sync + 'static
where
    F: Fn() -> Result<P> + Send + Sync + 'static,
{
    move |settings| {
        if let Some(setting) = settings.untaken() {
            return Err(setting.refuse(
                &format!("provider `{key}` was registered without settings"),
                "register it with `register_chat_with` (or its embedding and decision twins)",
            ));
        }
        constructor()
    }
}

#[cfg(test)]
mod tests {
    use allama::mock::{MockDecisionProvider, MockEmbeddingProvider, MockProvider, MockResponse};
    use allama::{ChatProvider, ChatRequest, EmbeddingProvider, EmbeddingRequest};

    use super::*;

    fn registry() -> Registry {
        let mut registry = Registry::new();
        registry
            .register_chat("mock", || Ok(MockProvider::new([MockResponse::text("hi")])))
            .register_embedding("mock", || {
                Ok(MockEmbeddingProvider::with_vectors(vec![vec![1.0]]))
            })
            .register_chat("chat-only", || Ok(MockProvider::empty()));
        registry
    }

    fn chat_error(registry: &Registry, address: &str) -> Error {
        match registry.chat_from_model(address) {
            Ok(_) => panic!("{address:?} resolved"),
            Err(err) => err,
        }
    }

    #[test]
    fn model_address_splits_at_first_colon() {
        let registry = registry();
        let (_, model) = registry.chat_from_model("mock:qwen3:1.7b").unwrap();
        assert_eq!(model, "qwen3:1.7b");
        let (_, model) = registry.chat_from_model("mock:a/b").unwrap();
        assert_eq!(model, "a/b");
    }

    #[test]
    fn malformed_model_address_is_configuration_error() {
        let registry = registry();
        for address in [
            "gpt-5.5",
            ":gpt-5.5",
            "mock:",
            "mock::",
            ":",
            "",
            // Whitespace is refused, never trimmed into another address.
            "mock: gpt-5",
            "mock:gpt-5 ",
            "mock:gpt-5\n",
            " mock:gpt-5",
            // Keys are not case-folded.
            "Mock:gpt-5",
        ] {
            let err = chat_error(&registry, address);
            assert!(
                matches!(err, Error::Configuration { .. }),
                "{address:?}: {err:?}"
            );
            assert!(err.message().contains("provider:model"), "{address:?}");
        }
    }

    #[test]
    fn a_near_miss_address_suggests_the_registered_one() {
        let registry = registry();
        for (address, suggestion) in [
            ("mock/gpt-5", "mock:gpt-5"),
            ("Mock:gpt-5", "mock:gpt-5"),
            ("mock: gpt-5", "mock:gpt-5"),
            ("mock:gpt-5\n", "mock:gpt-5"),
            ("mock/a/b", "mock:a/b"),
        ] {
            let err = chat_error(&registry, address);
            assert!(
                err.message()
                    .ends_with(&format!("did you mean `{suggestion}`?")),
                "{address:?}: {err}"
            );
        }
        // No hint for a key nobody registered.
        let err = chat_error(&registry, "meta-llama/llama-3");
        assert!(!err.message().contains("did you mean"), "{err}");
    }

    #[test]
    fn an_uppercase_key_suggests_the_registered_one() {
        let Err(err) = registry().chat_from_provider("Mock") else {
            panic!("`Mock` resolved");
        };
        assert!(err.message().contains("did you mean `mock`?"), "{err}");
    }

    #[tokio::test]
    async fn chat_from_model_returns_provider_and_model() {
        let (provider, model) = registry().chat_from_model("mock:some:model").unwrap();
        assert_eq!(model, "some:model");
        let response = provider
            .chat(&ChatRequest::new(model).user("hello"))
            .await
            .unwrap();
        assert_eq!(response.text_or_empty(), "hi");
    }

    #[tokio::test]
    async fn embedding_from_model_returns_provider_and_model() {
        let (provider, model) = registry().embedding_from_model("mock:embed-1").unwrap();
        assert_eq!(model, "embed-1");
        let response = provider
            .embed(&EmbeddingRequest::new(model).input("text"))
            .await
            .unwrap();
        assert_eq!(response.embeddings, vec![vec![1.0]]);
    }

    #[test]
    fn decision_from_model_returns_provider_and_model() {
        use allama::ProviderIdentity;

        let mut registry = registry();
        registry.register_decision("mock", || {
            Ok(MockDecisionProvider::empty().with_provider_name("mock-decision"))
        });
        let (provider, model) = registry.decision_from_model("mock:jev").unwrap();
        assert_eq!(provider.provider_name(), "mock-decision");
        assert_eq!(model, "jev");
    }

    #[test]
    fn a_key_without_the_operation_is_a_configuration_error() {
        let mut registry = registry();
        registry.register_decision("decide-only", || Ok(MockDecisionProvider::empty()));

        let err = registry.embedding_from_provider("chat-only").unwrap_err();
        assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
        assert_eq!(
            err.message(),
            "provider `chat-only` does not offer embeddings (it offers chat); \
             embedding providers: mock"
        );

        let err = registry.chat_from_provider("decide-only").unwrap_err();
        assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
        assert_eq!(
            err.message(),
            "provider `decide-only` does not offer chat (it offers decisions); \
             chat providers: chat-only, mock"
        );

        let err = registry.decision_from_provider("mock").unwrap_err();
        assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
        assert_eq!(
            err.message(),
            "provider `mock` does not offer decisions (it offers chat and embeddings); \
             decision providers: decide-only"
        );

        let err = Registry::new()
            .register_chat("only", || Ok(MockProvider::empty()))
            .decision_from_provider("only")
            .unwrap_err();
        assert_eq!(
            err.message(),
            "provider `only` does not offer decisions (it offers chat); \
             no decision providers are registered"
        );
    }

    #[test]
    fn an_unknown_key_lists_the_keys_that_offer_the_operation() {
        let mut registry = registry();
        registry.register_decision("decide-only", || Ok(MockDecisionProvider::empty()));

        let err = registry.chat_from_provider("nope").unwrap_err();
        assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
        assert_eq!(
            err.message(),
            "unknown chat provider `nope`; chat providers: chat-only, mock; \
             register your own with `Registry::register_chat`"
        );

        let err = registry.embedding_from_provider("nope").unwrap_err();
        assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
        assert_eq!(
            err.message(),
            "unknown embedding provider `nope`; embedding providers: mock; \
             register your own with `Registry::register_embedding`"
        );

        let err = registry.decision_from_provider("nope").unwrap_err();
        assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
        assert_eq!(
            err.message(),
            "unknown decision provider `nope`; decision providers: decide-only; \
             register your own with `Registry::register_decision`"
        );

        let err = Registry::new().chat_from_provider("nope").unwrap_err();
        assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
        assert_eq!(
            err.message(),
            "unknown chat provider `nope`; no chat providers are registered; \
             register your own with `Registry::register_chat`"
        );
    }

    #[test]
    fn constructor_errors_pass_through() {
        let mut registry = Registry::new();
        registry.register_chat("broken", || {
            Err::<MockProvider, _>(Error::configuration("BROKEN_API_KEY not set"))
        });
        let err = registry.chat_from_provider("broken").unwrap_err();
        assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
    }

    #[test]
    fn registering_replaces_the_previous_constructor() {
        let mut registry = registry();
        registry.register_chat("chat-only", || {
            Err::<MockProvider, _>(Error::auth("replaced"))
        });
        let err = registry.chat_from_provider("chat-only").unwrap_err();
        assert_eq!(err.message(), "replaced");
    }

    #[test]
    #[should_panic(expected = "invalid provider key")]
    fn key_with_colon_panics() {
        Registry::new().register_chat("a:b", || Ok(MockProvider::empty()));
    }

    #[test]
    fn keys_outside_the_grammar_panic() {
        for key in ["", "OpenAI ", "OpenAI", "a/b", "-a", "a_b", "a b"] {
            let result = std::panic::catch_unwind(|| {
                Registry::new().register_chat(key, || Ok(MockProvider::empty()));
            });
            assert!(result.is_err(), "{key:?} was registered");
        }
        let mut registry = Registry::new();
        registry.register_chat("my-gateway2", || Ok(MockProvider::empty()));
        registry.register_chat("0x", || Ok(MockProvider::empty()));
        assert_eq!(registry.keys().collect::<Vec<_>>(), ["0x", "my-gateway2"]);
    }

    #[test]
    fn settings_reach_the_constructor() {
        use allama::ProviderIdentity;

        let name = |settings: &mut AdapterSettings| {
            settings
                .take_credential_scope()
                .unwrap_or_else(|| "default".into())
        };
        let mut registry = Registry::new();
        registry
            .register_chat_with("scoped", move |settings| {
                Ok(MockProvider::empty().with_provider_name(name(settings)))
            })
            .register_embedding_with("scoped", move |settings| {
                Ok(MockEmbeddingProvider::with_vectors(vec![]).with_provider_name(name(settings)))
            })
            .register_decision_with("scoped", move |settings| {
                Ok(MockDecisionProvider::empty().with_provider_name(name(settings)))
            });
        let settings = ProviderSettings::new().credential_scope("tenant-a");

        let (chat, model) = registry
            .chat_from_model_with("scoped:m", &settings)
            .unwrap();
        assert_eq!((chat.provider_name(), model.as_str()), ("tenant-a", "m"));
        assert_eq!(
            registry
                .chat_from_provider("scoped")
                .unwrap()
                .provider_name(),
            "default"
        );
        let (embedding, _) = registry
            .embedding_from_model_with("scoped:e", &settings)
            .unwrap();
        assert_eq!(embedding.provider_name(), "tenant-a");
        let (decision, _) = registry
            .decision_from_model_with("scoped:d", &settings)
            .unwrap();
        assert_eq!(decision.provider_name(), "tenant-a");
        assert!(matches!(
            registry.chat_from_provider_with("nope", &settings),
            Err(Error::Configuration { .. })
        ));
    }

    /// A registry whose constructors take settings and ignore them all.
    fn ignoring() -> Registry {
        let mut registry = Registry::new();
        registry
            .register_chat_with("ignoring", |_: &mut AdapterSettings| {
                Ok(MockProvider::empty())
            })
            .register_embedding_with("ignoring", |_: &mut AdapterSettings| {
                Ok(MockEmbeddingProvider::with_vectors(vec![]))
            })
            .register_decision_with("ignoring", |_: &mut AdapterSettings| {
                Ok(MockDecisionProvider::empty())
            });
        registry
    }

    #[test]
    fn a_constructor_that_ignores_a_scope_is_refused() {
        let registry = ignoring();
        let settings = ProviderSettings::new().credential_scope("tenant-a");
        for err in [
            registry.chat_from_model_with("ignoring:m", &settings).err(),
            registry
                .embedding_from_model_with("ignoring:e", &settings)
                .err(),
            registry
                .decision_from_model_with("ignoring:d", &settings)
                .err(),
        ] {
            let err = err.expect("a scope the constructor did not take fails the lookup");
            assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
            assert_eq!(
                err.message(),
                "the constructor of provider `ignoring` did not take a credential scope, so the \
                 provider was built without it; take it with \
                 `AdapterSettings::take_credential_scope` and apply it with the adapter's \
                 `with_credential_scope`"
            );
        }
        // Without adapter settings, or with only stack settings, it builds.
        for settings in [ProviderSettings::new(), quick_retries()] {
            assert!(
                registry
                    .chat_from_model_with("ignoring:m", &settings)
                    .is_ok()
            );
            assert!(
                registry
                    .embedding_from_model_with("ignoring:e", &settings)
                    .is_ok()
            );
            assert!(
                registry
                    .decision_from_model_with("ignoring:d", &settings)
                    .is_ok()
            );
        }
    }

    #[test]
    fn a_chat_constructor_that_ignores_capabilities_is_refused() {
        use allama::{CapabilitySupport, ChatCapability};

        let registry = ignoring();
        let settings = ProviderSettings::new()
            .chat_capabilities(|_: &str, _: ChatCapability| Some(CapabilitySupport::Supported));
        let err = registry
            .chat_from_model_with("ignoring:m", &settings)
            .expect_err("overrides the constructor did not take fail the lookup");
        assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
        assert!(
            err.message()
                .contains("`AdapterSettings::take_chat_capabilities`"),
            "{}",
            err.message()
        );
        // Chat answers are not handed to the other operations at all.
        assert!(
            registry
                .embedding_from_model_with("ignoring:e", &settings)
                .is_ok()
        );
        assert!(
            registry
                .decision_from_model_with("ignoring:d", &settings)
                .is_ok()
        );
    }

    #[test]
    fn a_constructor_error_wins_over_an_untaken_setting() {
        let mut registry = Registry::new();
        registry.register_chat_with("broken", |_: &mut AdapterSettings| {
            Err::<MockProvider, _>(Error::configuration("BROKEN_API_KEY not set"))
        });
        let settings = ProviderSettings::new().credential_scope("tenant-a");
        let err = registry
            .chat_from_provider_with("broken", &settings)
            .unwrap_err();
        assert_eq!(err.message(), "BROKEN_API_KEY not set");
    }

    #[test]
    fn a_constructor_without_settings_is_not_run_with_a_scope() {
        let runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut registry = Registry::new();
        registry.register_chat("plain", {
            let runs = runs.clone();
            move || {
                runs.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Ok(MockProvider::empty())
            }
        });
        let settings = ProviderSettings::new().credential_scope("tenant-a");
        assert!(
            registry
                .chat_from_provider_with("plain", &settings)
                .is_err()
        );
        assert_eq!(runs.load(std::sync::atomic::Ordering::Relaxed), 0);
    }

    #[test]
    fn a_constructor_without_settings_refuses_a_scope() {
        let registry = registry();
        let settings = ProviderSettings::new().credential_scope("tenant-a");
        for err in [
            registry.chat_from_model_with("mock:m", &settings).err(),
            registry
                .embedding_from_model_with("mock:m", &settings)
                .err(),
        ] {
            let err = err.expect("a scoped lookup of a constructor without settings fails");
            assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
            assert!(err.message().contains("`mock`"), "{}", err.message());
            assert!(
                err.message().contains("register_chat_with"),
                "{}",
                err.message()
            );
        }
        // The defaults still build it.
        assert!(
            registry
                .chat_from_model_with("mock:m", &ProviderSettings::new())
                .is_ok()
        );
    }

    /// The second error each mock below returns: not retryable, so a
    /// retried call ends with it.
    fn last_error() -> Error {
        Error::invalid_request("final")
    }

    fn quick_retries() -> ProviderSettings {
        ProviderSettings::new().retry(Some(
            allama::wrappers::RetryPolicy::new()
                .base_delay(std::time::Duration::from_millis(1))
                .jitter(false),
        ))
    }

    #[tokio::test]
    async fn every_operation_retries_by_default() {
        use allama::DecisionProvider;
        use allama::decision::{DecisionRequest, Question};

        let chat = MockProvider::build(|b| b.error(Error::overloaded("busy")).error(last_error()));
        let embedding =
            MockEmbeddingProvider::new([Err(Error::overloaded("busy")), Err(last_error())]);
        let decision =
            MockDecisionProvider::new([Err(Error::overloaded("busy")), Err(last_error())]);
        let mut registry = Registry::new();
        registry
            .register_chat("mock", {
                let chat = chat.clone();
                move || Ok(chat.clone())
            })
            .register_embedding("mock", {
                let embedding = embedding.clone();
                move || Ok(embedding.clone())
            })
            .register_decision("mock", {
                let decision = decision.clone();
                move || Ok(decision.clone())
            });

        // The defaults, as `chat_from_model` uses them.
        let (provider, model) = registry.chat_from_model("mock:m").unwrap();
        let err = provider
            .chat(&ChatRequest::new(model).user("hi"))
            .await
            .unwrap_err();
        assert_eq!(err.message(), "final");
        assert_eq!(chat.call_count(), 2, "the overload was retried");

        let (provider, model) = registry.embedding_from_model("mock:e").unwrap();
        let err = provider
            .embed(&EmbeddingRequest::new(model).input("text"))
            .await
            .unwrap_err();
        assert_eq!(err.message(), "final");
        assert_eq!(embedding.requests().len(), 2);

        let (provider, model) = registry.decision_from_model("mock:d").unwrap();
        let request = DecisionRequest::new(model, "x").question("q", Question::yes_no("?"));
        let err = provider.decide(&request).await.unwrap_err();
        assert_eq!(err.message(), "final");
        assert_eq!(decision.requests().len(), 2);
    }

    #[tokio::test]
    async fn retries_follow_the_settings() {
        let chat = MockProvider::build(|b| b.error(Error::overloaded("busy")).text("hi"));
        let mut registry = Registry::new();
        registry.register_chat("mock", {
            let chat = chat.clone();
            move || Ok(chat.clone())
        });

        // Off: the overload reaches the caller.
        let settings = ProviderSettings::new().retry(None);
        let (provider, model) = registry.chat_from_model_with("mock:m", &settings).unwrap();
        let request = ChatRequest::new(model).user("hi");
        let err = provider.chat(&request).await.unwrap_err();
        assert!(matches!(err, Error::Overloaded { .. }), "{err:?}");

        // The scripted answer the call above did not reach.
        assert_eq!(provider.chat(&request).await.unwrap().text_or_empty(), "hi");

        // A policy of its own, with a shared budget the retry draws on (a
        // retry that fails keeps its cost).
        chat.push_response(Error::overloaded("busy again"));
        chat.push_response(last_error());
        let budget = allama::wrappers::RetryBudget::new();
        let mut settings = quick_retries();
        settings.retry = settings.retry.map(|policy| policy.budget(budget.clone()));
        let (provider, _) = registry.chat_from_model_with("mock:m", &settings).unwrap();
        let err = provider.chat(&request).await.unwrap_err();
        assert_eq!(err.message(), "final");
        assert!(budget.available() < allama::wrappers::RetryBudget::DEFAULT_CAPACITY);
    }

    #[tokio::test]
    async fn attempt_timeout_bounds_each_attempt_inside_the_retries() {
        use std::time::Duration;

        use allama::wrappers::TimeoutPolicy;

        let chat = MockProvider::build(|b| {
            b.response(MockResponse::delayed(
                Duration::from_secs(10),
                MockResponse::text("too late"),
            ))
            .text("hi")
        });
        let mut registry = Registry::new();
        registry.register_chat("mock", {
            let chat = chat.clone();
            move || Ok(chat.clone())
        });
        let settings = quick_retries().attempt_timeout(Some(
            TimeoutPolicy::new().request(Duration::from_millis(20)),
        ));
        let (provider, model) = registry.chat_from_model_with("mock:m", &settings).unwrap();

        // The stalled attempt times out and is retried.
        let response = provider
            .chat(&ChatRequest::new(model).user("hi"))
            .await
            .unwrap();
        assert_eq!(response.text_or_empty(), "hi");
        assert_eq!(chat.call_count(), 2);
    }

    #[tokio::test]
    async fn unsupported_policy_fills_requests_without_one() {
        use allama::UnsupportedPolicy;

        let chat = MockProvider::build(|b| b.text("a").text("b"));
        let embedding = MockEmbeddingProvider::with_vectors(vec![vec![1.0]]);
        let mut registry = Registry::new();
        registry
            .register_chat("mock", {
                let chat = chat.clone();
                move || Ok(chat.clone())
            })
            .register_embedding("mock", {
                let embedding = embedding.clone();
                move || Ok(embedding.clone())
            });
        let settings = ProviderSettings::new().unsupported_policy(UnsupportedPolicy::Error);

        let (provider, model) = registry.chat_from_model_with("mock:m", &settings).unwrap();
        let request = ChatRequest::new(model).user("hi");
        provider.chat(&request).await.unwrap();
        provider
            .chat(&request.clone().unsupported_policy(UnsupportedPolicy::Warn))
            .await
            .unwrap();
        let policies: Vec<_> = chat
            .requests()
            .iter()
            .map(|request| request.unsupported_policy)
            .collect();
        assert_eq!(
            policies,
            [
                Some(UnsupportedPolicy::Error),
                Some(UnsupportedPolicy::Warn)
            ]
        );

        let (provider, model) = registry
            .embedding_from_model_with("mock:e", &settings)
            .unwrap();
        provider
            .embed(&EmbeddingRequest::new(model).input("text"))
            .await
            .unwrap();
        assert_eq!(
            embedding.requests()[0].unsupported_policy,
            Some(UnsupportedPolicy::Error)
        );
    }

    #[test]
    fn chat_capabilities_reach_a_constructor_with_settings() {
        use allama::{CapabilitySupport, ChatCapability};

        let mut registry = Registry::new();
        registry.register_chat_with("mock", |settings: &mut AdapterSettings| {
            let mut provider = MockProvider::empty();
            if let Some(resolver) = settings.take_chat_capabilities() {
                let answer = resolver.chat_capability("m", ChatCapability::ImageInput);
                provider =
                    provider.with_chat_capability(ChatCapability::ImageInput, answer.unwrap());
            }
            Ok(provider)
        });
        let settings = ProviderSettings::new().chat_capabilities(|_: &str, capability| {
            (capability == ChatCapability::ImageInput).then_some(CapabilitySupport::Supported)
        });

        let (provider, model) = registry.chat_from_model_with("mock:m", &settings).unwrap();
        assert!(
            provider
                .chat_capability(&model, ChatCapability::ImageInput)
                .is_supported()
        );
    }

    #[test]
    fn a_constructor_without_settings_refuses_chat_capabilities_only_for_chat() {
        use allama::{CapabilitySupport, ChatCapability};

        let registry = registry();
        let settings = ProviderSettings::new()
            .chat_capabilities(|_: &str, _: ChatCapability| Some(CapabilitySupport::Supported));
        let err = registry
            .chat_from_model_with("mock:m", &settings)
            .expect_err("a chat constructor without settings cannot apply them");
        assert!(matches!(err, Error::Configuration { .. }), "{err:?}");
        assert!(
            err.message().contains("chat capability overrides"),
            "{}",
            err.message()
        );
        // They are chat answers, so an embedding lookup builds.
        assert!(
            registry
                .embedding_from_model_with("mock:e", &settings)
                .is_ok()
        );
        // So do the wrapper settings.
        assert!(
            registry
                .chat_from_model_with("mock:m", &quick_retries())
                .is_ok()
        );
    }

    #[test]
    fn settings_debug_lists_every_setting() {
        let debug = format!(
            "{:?}",
            ProviderSettings::new()
                .credential_scope("tenant-a")
                .chat_capabilities(|_: &str, _: allama::ChatCapability| None)
        );
        for field in [
            "credential_scope: Some(\"tenant-a\")",
            "chat_capabilities: Some(\"<resolver>\")",
            "unsupported_policy: None",
            "attempt_timeout: None",
            "retry: Some(RetryPolicy { max_attempts: 3",
        ] {
            assert!(debug.contains(field), "{field} in {debug}");
        }
    }
}
