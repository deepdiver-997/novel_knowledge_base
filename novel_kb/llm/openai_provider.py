from novel_kb.llm.models import AnalysisResult, EmbeddingResult
from novel_kb.llm.provider import LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(self, config) -> None:
        self.config = config

    def analyze_characters(self, text: str) -> AnalysisResult:
        raise NotImplementedError("OpenAIProvider is not implemented yet")

    def analyze_plot(self, text: str) -> AnalysisResult:
        raise NotImplementedError("OpenAIProvider is not implemented yet")

    def extract_features(self, text: str) -> AnalysisResult:
        raise NotImplementedError("OpenAIProvider is not implemented yet")

    def generate_embedding(self, text: str) -> EmbeddingResult:
        raise NotImplementedError("OpenAIProvider is not implemented yet")

    def health_check(self) -> bool:
        return False
