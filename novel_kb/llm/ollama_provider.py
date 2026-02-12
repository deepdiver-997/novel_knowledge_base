import json
from typing import Any, Dict

import requests

from novel_kb.llm.models import AnalysisResult, EmbeddingResult
from novel_kb.llm.provider import LLMProvider


class OllamaProvider(LLMProvider):
    def __init__(self, config) -> None:
        self.base_url = config.ollama_base_url.rstrip("/")
        self.model = config.ollama_model
        self.temperature = config.temperature
        self.timeout = config.timeout

    def analyze_characters(self, text: str) -> AnalysisResult:
        prompt = (
            "Extract character relationships as JSON with keys: "
            "characters (array of {name, role, relationships})."
        )
        return self._analyze(prompt, text, kind="characters")

    def analyze_plot(self, text: str) -> AnalysisResult:
        prompt = "Summarize plot in JSON with key: summary."
        return self._analyze(prompt, text, kind="plot")

    def extract_features(self, text: str) -> AnalysisResult:
        prompt = "Extract feature tags in JSON with key: features (array)."
        return self._analyze(prompt, text, kind="features")

    def generate_embedding(self, text: str) -> EmbeddingResult:
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        vector = payload.get("embedding", [])
        return EmbeddingResult(vector=vector, tokens_used=0)

    def health_check(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _analyze(self, prompt: str, text: str, kind: str) -> AnalysisResult:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": f"{prompt}\n\n{text}",
                "temperature": self.temperature,
                "format": "json",
                "stream": False,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
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
