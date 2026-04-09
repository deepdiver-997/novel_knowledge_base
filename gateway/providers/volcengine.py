import json
import os
from typing import Any, Dict, Optional

from gateway.models import AnalysisResult, EmbeddingResult
from gateway.providers.base import LLMProvider

try:
    from volcenginesdkarkruntime import Ark
except ImportError:
    Ark = None


class VolcEngineProvider(LLMProvider):
    """VolcEngine Ark Provider (OpenAI-compatible style)."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
        model: str = "doubao-1-5-pro-32k-250115",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 60,
    ) -> None:
        self.api_key = api_key or os.getenv("ARK_API_KEY")
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max(1, min(int(max_tokens), 8000))
        self.timeout = timeout

        if not Ark:
            raise RuntimeError("volcengine-ark-runtime is required")
        if not self.api_key:
            raise RuntimeError("ARK_API_KEY is required")

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
            provider_type="volcengine",
            model_name=self.model,
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
