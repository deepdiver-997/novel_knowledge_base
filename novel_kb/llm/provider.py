from abc import ABC, abstractmethod
from typing import List

from novel_kb.llm.models import AnalysisResult, EmbeddingResult


class LLMProvider(ABC):
    @abstractmethod
    def analyze_characters(self, text: str) -> AnalysisResult:
        raise NotImplementedError

    @abstractmethod
    def analyze_plot(self, text: str) -> AnalysisResult:
        raise NotImplementedError

    @abstractmethod
    def extract_features(self, text: str) -> AnalysisResult:
        raise NotImplementedError

    @abstractmethod
    def generate_embedding(self, text: str) -> EmbeddingResult:
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> bool:
        raise NotImplementedError
