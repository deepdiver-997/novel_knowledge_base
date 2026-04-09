import json
from typing import Any, Dict

import aiohttp

from novel_kb.llm.models import AnalysisResult, EmbeddingResult
from novel_kb.llm.provider import LLMProvider


class OllamaProvider(LLMProvider):
    def __init__(self, config) -> None:
        self.base_url = config.ollama_base_url.rstrip("/")
        self.model = config.ollama_model
        self.embedding_model = config.ollama_embedding_model or self.model
        self.temperature = config.temperature
        self.timeout = config.timeout

    async def analyze_characters(self, text: str) -> AnalysisResult:
        prompt = (
            "Return JSON with key 'characters' (array of {name, role, relationships}). "
            "Use Simplified Chinese for all text values. Keep JSON keys in English."
        )
        return await self._analyze(prompt, text, kind="characters")

    async def analyze_plot(self, text: str) -> AnalysisResult:
        prompt = "Return JSON with key 'summary'. Use Simplified Chinese for the summary text."
        return await self._analyze(prompt, text, kind="plot")

    async def extract_features(self, text: str) -> AnalysisResult:
        prompt = (
            "Return JSON with key 'features' (array). "
            "Use Simplified Chinese for feature text. Keep JSON keys in English."
        )
        return await self._analyze(prompt, text, kind="features")

    async def generate_embedding(self, text: str) -> EmbeddingResult:
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.embedding_model, "prompt": text},
            ) as response:
                response.raise_for_status()
                payload = await response.json()
                vector = payload.get("embedding", [])
                return EmbeddingResult(vector=vector, tokens_used=0)

    async def health_check(self) -> bool:
        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    return response.status == 200
        except aiohttp.ClientError:
            return False

    async def _analyze(self, prompt: str, text: str, kind: str) -> AnalysisResult:
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": f"{prompt}\n\n{text}",
                    "temperature": self.temperature,
                    "format": "json",
                    "stream": False,
                },
            ) as response:
                response.raise_for_status()
                payload = await response.json()
                raw = payload.get("response", "{}")
                return AnalysisResult(kind=kind, content=self._safe_json(raw), tokens_used=0)

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

