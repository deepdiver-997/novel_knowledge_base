from abc import ABC, abstractmethod
from typing import Optional

from gateway.models import AnalysisResult, EmbeddingResult


class LLMProvider(ABC):
    @abstractmethod
    async def analyze_characters(self, text: str) -> AnalysisResult:
        raise NotImplementedError

    @abstractmethod
    async def analyze_plot(self, text: str) -> AnalysisResult:
        raise NotImplementedError

    @abstractmethod
    async def extract_features(self, text: str) -> AnalysisResult:
        raise NotImplementedError

    @abstractmethod
    async def generate_embedding(self, text: str) -> EmbeddingResult:
        raise NotImplementedError

    @abstractmethod
    async def health_check(self) -> bool:
        raise NotImplementedError

    def configure_limits(self, concurrency_limit: int, qps_limit: float) -> None:
        return None
