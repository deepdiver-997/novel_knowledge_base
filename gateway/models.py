from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class AnalysisResult:
    kind: str
    content: Dict[str, Any]
    tokens_used: int = 0
    provider_type: Optional[str] = None
    model_name: Optional[str] = None


@dataclass
class EmbeddingResult:
    vector: List[float]
    tokens_used: int = 0


@dataclass
class GatewayResponse:
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    latency_ms: int = 0


@dataclass
class HealthStatus:
    healthy: bool
    providers: Dict[str, bool]
    active_tiers: List[str]
