import json
from typing import Any, Dict, Optional

from novel_kb.llm.models import AnalysisResult, EmbeddingResult
from novel_kb.llm.provider import LLMProvider
from novel_kb.utils.errors import ProviderError

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover - optional dependency
    AsyncOpenAI = None


class XunfeiProvider(LLMProvider):
    def __init__(self, config) -> None:
        self.api_key = config.xunfei_api_key
        self.base_url = config.xunfei_base_url.rstrip("/")
        self.model = config.xunfei_model
        self.embedding_model = getattr(config, "xunfei_embedding_model", "")
        self.lora_id = getattr(config, "xunfei_lora_id", "")
        self.temperature = config.temperature
        # 处理 max_tokens 列表配置，并限制到可用范围
        if isinstance(config.max_tokens, list):
            self.max_tokens = config.max_tokens[0] if config.max_tokens else 2048
        else:
            self.max_tokens = config.max_tokens
        self.max_tokens = max(1, min(int(self.max_tokens), 8000))
        self.timeout = config.timeout
        self.client: Optional[Any] = None
        if AsyncOpenAI and self.api_key:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )

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
        if not self.embedding_model:
            return EmbeddingResult(vector=[], tokens_used=0)
        client = self._require_client()
        response = await client.embeddings.create(model=self.embedding_model, input=text)
        vector = []
        if response.data:
            vector = response.data[0].embedding
        return EmbeddingResult(vector=vector, tokens_used=self._tokens_used(response))

    async def health_check(self) -> bool:
        if not self.client:
            return False
        try:
            await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                stream=False,
                temperature=0,
                max_tokens=1,
            )
            return True
        except Exception:
            return False

    async def _analyze(self, prompt: str, text: str, kind: str) -> AnalysisResult:
        client = self._require_client()
        extra_headers = {"lora_id": self.lora_id} if self.lora_id else None
        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You output strict JSON."},
                {"role": "user", "content": f"{prompt}\n\n{text}"},
            ],
            stream=False,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_headers=extra_headers,
            extra_body={
                "response_format": {"type": "json_object"},
                "search_disable": True,
            },
        )
        message = response.choices[0].message
        raw = message.content or "{}"
        return AnalysisResult(kind=kind, content=self._safe_json(raw), tokens_used=self._tokens_used(response))

    def _require_client(self) -> Any:
        if not self.client:
            raise ProviderError("Xunfei client is not configured")
        return self.client

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
