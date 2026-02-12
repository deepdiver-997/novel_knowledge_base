from novel_kb.llm.provider import LLMProvider


class CharacterAnalyzer:
    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    def analyze(self, text: str):
        return self.provider.analyze_characters(text)
