import json
import os
from typing import Any, Dict, Optional

from novel_kb.llm.models import AnalysisResult, EmbeddingResult
from novel_kb.llm.provider import LLMProvider
from novel_kb.utils.errors import ProviderError, ProviderErrorDecision, ProviderFailureAction

try:
    from volcenginesdkarkruntime import Ark
except ImportError:  # pragma: no cover - optional dependency
    Ark = None


class VolcEngineProvider(LLMProvider):
    """VolcEngine Ark Provider (OpenAI-compatible style)."""

    def __init__(self, config) -> None:
        self.api_key = getattr(config, "volcengine_api_key", None) or os.getenv("ARK_API_KEY")
        self.base_url = getattr(
            config,
            "volcengine_base_url",
            "https://ark.cn-beijing.volces.com/api/v3",
        )
        self.model = getattr(config, "volcengine_model", None) or "doubao-1-5-pro-32k-250115"
        self.temperature = config.temperature
        if isinstance(config.max_tokens, list):
            self.max_tokens = config.max_tokens[0] if config.max_tokens else 2048
        else:
            self.max_tokens = config.max_tokens
        self.timeout = config.timeout

        if not Ark:
            raise ProviderError("volcengine-ark-runtime is required")
        if not self.api_key:
            raise ProviderError("ARK_API_KEY is required")

        self.client = Ark(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    async def analyze_characters(self, text: str) -> AnalysisResult:
        prompt = (
            "Extract character relationships as JSON with keys: "
            "characters (array of {name, role, relationships})."
        )
        return await self._analyze(prompt, text, kind="characters")

    async def analyze_plot(self, text: str) -> AnalysisResult:
        prompt = "Summarize plot in JSON with key: summary."
        return await self._analyze(prompt, text, kind="plot")

    async def extract_features(self, text: str) -> AnalysisResult:
        prompt = "Extract feature tags in JSON with key: features (array)."
        return await self._analyze(prompt, text, kind="features")

    async def generate_embedding(self, text: str) -> EmbeddingResult:
        return EmbeddingResult(vector=[], tokens_used=0)

    async def health_check(self) -> bool:
        try:
            await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                temperature=0,
                max_tokens=10,
            )
            return True
        except Exception:
            return False

    def classify_error(self, exc: Exception) -> Optional[ProviderErrorDecision]:
        message = str(exc).lower()
        if "allocationquota" in message or "quota" in message or "insufficient" in message:
            return ProviderErrorDecision(
                action=ProviderFailureAction.DISABLE,
                reason="quota_exhausted",
            )
        if "429" in message or "rate limit" in message or "too many requests" in message:
            return ProviderErrorDecision(
                action=ProviderFailureAction.THROTTLE,
                reason="rate_limited",
            )
        if "timeout" in message or "timed out" in message or "connection" in message:
            return ProviderErrorDecision(
                action=ProviderFailureAction.THROTTLE,
                reason="network_error",
            )
        return None

    async def _analyze(self, prompt: str, text: str, kind: str) -> AnalysisResult:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You output strict JSON."},
                {"role": "user", "content": f"{prompt}\n\n{text}"},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        message = response.choices[0].message
        raw = message.content or "{}"
        return AnalysisResult(
            kind=kind,
            content=self._safe_json(raw),
            tokens_used=self._tokens_used(response),
        )

    @staticmethod
    def _tokens_used(response: Any) -> int:
        usage = getattr(response, "usage", None)
        total = getattr(usage, "total_tokens", None) if usage else None
        return total if isinstance(total, int) else 0

    @staticmethod
    def _safe_json(raw: str) -> Dict[str, Any]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            cleaned = raw.strip()
            if cleaned.startswith("{") and cleaned.endswith("}"):
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    pass
            if "{" in cleaned and "}" in cleaned:
                start = cleaned.find("{")
                end = cleaned.rfind("}")
                if start < end:
                    snippet = cleaned[start:end + 1]
                    try:
                        return json.loads(snippet)
                    except json.JSONDecodeError:
                        pass
            return {"raw": raw}
