from novel_kb.llm.models import AnalysisResult, EmbeddingResult
from novel_kb.llm.provider import LLMProvider


class NoOpProvider(LLMProvider):
    def analyze_characters(self, text: str) -> AnalysisResult:
        return AnalysisResult(kind="characters", content={"characters": []}, tokens_used=0)

    def analyze_plot(self, text: str) -> AnalysisResult:
        return AnalysisResult(kind="plot", content={"summary": ""}, tokens_used=0)

    def extract_features(self, text: str) -> AnalysisResult:
        return AnalysisResult(kind="features", content={"features": []}, tokens_used=0)

    def generate_embedding(self, text: str) -> EmbeddingResult:
        return EmbeddingResult(vector=[], tokens_used=0)

    def health_check(self) -> bool:
        return False
