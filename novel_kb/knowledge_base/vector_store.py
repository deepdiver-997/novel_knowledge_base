import math
from typing import Dict, List, Tuple

from novel_kb.knowledge_base.models import VectorRecord


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._records: List[VectorRecord] = []

    def add(self, record: VectorRecord) -> None:
        self._records.append(record)

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Tuple[VectorRecord, float]]:
        scored = []
        for record in self._records:
            score = self._cosine_similarity(query_vector, record.vector)
            scored.append((record, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
