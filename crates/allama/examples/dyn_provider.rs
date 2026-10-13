//! Implement a provider and use it through `DynChatProvider`.

use allama::prelude::*;
use allama::{
    CapabilitySupport, ChatCapability, DynChatProvider, FinishReason, ProviderIdentity,
    SingleResponseStream,
};

struct StaticProvider;

impl StaticProvider {
    const PROVIDER_NAME: &str = "static-demo";
}

impl ProviderIdentity for StaticProvider {
    fn provider_name(&self) -> &str {
        Self::PROVIDER_NAME
    }

    /// It writes no provider data, so it reads none.
    fn reads_namespace(&self, _namespace: &str) -> bool {
        false
    }

    fn downcast_any(&self, id: std::any::TypeId) -> Option<&(dyn std::any::Any + Send + Sync)> {
        allama::expose(self, id)
    }
}

impl ChatProvider for StaticProvider {
    type Stream = SingleResponseStream;

    async fn chat(&self, request: &ChatRequest) -> allama::Result<ChatResponse> {
        Ok(ChatResponse::new(vec![ContentBlock::text(format!(
            "hello from {}",
            request.model
        ))])
        .finish_reason(FinishReason::Stop)
        .model(request.model.clone())
        .provider(Self::PROVIDER_NAME))
    }

    async fn chat_stream(&self, request: &ChatRequest) -> allama::Result<Self::Stream> {
        Ok(self.chat(request).await?.into_stream())
    }

    fn chat_capability(&self, _model: &str, capability: ChatCapability) -> CapabilitySupport {
        match capability {
            ChatCapability::Streaming => CapabilitySupport::Supported,
            ChatCapability::NativeStreaming => CapabilitySupport::Unsupported,
            _ => CapabilitySupport::Unknown,
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let request = ChatRequest::new("demo-model")
        .system("You are concise.")
        .user("Say hello");
    let provider = DynChatProvider::new(StaticProvider);

    let response = provider.chat(&request).await?;
    println!("dyn chat text: {}", response.text_or_empty());

    let streamed = provider
        .chat_stream(&request)
        .await?
        .collect_response()
        .await?;
    println!("dyn stream text: {}", streamed.text_or_empty());
    Ok(())
}
