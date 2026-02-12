from novel_kb.llm.aliyun_provider import AliyunProvider
from novel_kb.llm.ollama_provider import OllamaProvider
from novel_kb.llm.openai_provider import OpenAIProvider
from novel_kb.llm.provider import LLMProvider


class LLMFactory:
    _providers = {
        "ollama": OllamaProvider,
        "openai": OpenAIProvider,
        "aliyun": AliyunProvider,
    }

    @classmethod
    def create(cls, config) -> LLMProvider:
        provider_class = cls._providers.get(config.provider)
        if provider_class is None:
            raise ValueError(f"Unknown provider: {config.provider}")
        provider = provider_class(config)
        return provider

    @classmethod
    def register_provider(cls, name: str, provider_class: type) -> None:
        cls._providers[name] = provider_class
