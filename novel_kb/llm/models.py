from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class AnalysisResult:
    kind: str
    content: Dict[str, Any]
    tokens_used: int = 0


@dataclass
class EmbeddingResult:
    vector: List[float]
    tokens_used: int = 0
