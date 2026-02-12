from dataclasses import dataclass
from typing import Dict, List


@dataclass
class VectorRecord:
    record_id: str
    vector: List[float]
    metadata: Dict[str, str]
