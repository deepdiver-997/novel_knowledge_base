from novel_kb.llm.provider import LLMProvider


class FeatureExtractor:
    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    def extract(self, text: str):
        return self.provider.extract_features(text)
