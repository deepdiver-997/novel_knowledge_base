from novel_kb.llm.provider import LLMProvider


class EmbeddingBuilder:
    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    def build(self, text: str):
        return self.provider.generate_embedding(text)
