import json
from typing import Any, Dict, List, Optional, cast

from gateway.models import AnalysisResult, EmbeddingResult
from gateway.providers.base import LLMProvider

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None


class OpenAILikeProvider(LLMProvider):
    """统一的 OpenAI 兼容协议 Provider"""

    def __init__(
        self,
        provider_name: str,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 60,
        embedding_model: str = "",
        extra_headers: Optional[Dict[str, str]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.provider_name = provider_name
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max(1, min(int(max_tokens), 8000))
        self.timeout = timeout
        self.embedding_model = embedding_model
        self.extra_headers = extra_headers or None
        self.extra_body = extra_body or None
        self.client: Optional[Any] = None

        if AsyncOpenAI and self.api_key:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )

    async def analyze_characters(self, text: str) -> AnalysisResult:
        prompt = (
            "Return JSON with key 'characters' (array of {name, role, relationships}). "
            "Use Simplified Chinese for all text values. Keep JSON keys in English."
        )
        return await self._analyze(prompt, text, kind="characters")

    async def analyze_plot(self, text: str) -> AnalysisResult:
        prompt = (
            "Return JSON with key 'summary'. "
            "Only summarize the given input text itself in Simplified Chinese. "
            "Do not ask for more input. Do not output self-introduction. "
            "If the input lacks usable story content, return {'summary': ''}."
        )
        return await self._analyze(prompt, text, kind="plot")

    async def extract_features(self, text: str) -> AnalysisResult:
        prompt = (
            "Return JSON with key 'features' (array). "
            "Use Simplified Chinese for feature text. Keep JSON keys in English."
        )
        return await self._analyze(prompt, text, kind="features")

    async def generate_embedding(self, text: str) -> EmbeddingResult:
        if not self.embedding_model:
            return EmbeddingResult(vector=[], tokens_used=0)
        client = self._require_client()
        response = await client.embeddings.create(model=self.embedding_model, input=text)
        vector: List[float] = []
        if response.data:
            raw_vector = response.data[0].embedding
            if isinstance(raw_vector, list):
                for item in cast(List[Any], raw_vector):
                    try:
                        vector.append(float(item))
                    except (TypeError, ValueError):
                        continue
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
                max_tokens=10,
            )
            return True
        except Exception:
            return False

    async def _analyze(self, prompt: str, text: str, kind: str) -> AnalysisResult:
        client = self._require_client()
        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You output strict JSON only. "
                        "All natural language content must be in Simplified Chinese. "
                        "Keep JSON keys in English. "
                        "Never ask the user for more information. "
                        "Never output self-introduction or generic assistant disclaimers."
                    ),
                },
                {"role": "user", "content": f"{prompt}\n\nInputText:\n{text}"},
            ],
            stream=False,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_headers=self.extra_headers,
            extra_body=self.extra_body,
        )
        raw = self._extract_text_content(response)
        return AnalysisResult(
            kind=kind,
            content=self._safe_json(raw),
            tokens_used=self._tokens_used(response),
            provider_type=self.provider_name,
            model_name=self.model,
        )

    def _require_client(self) -> Any:
        if not self.client:
            raise RuntimeError(f"{self.provider_name} client is not configured")
        return self.client

    @staticmethod
    def _tokens_used(response: Any) -> int:
        if isinstance(response, dict):
            usage = response.get("usage")
            if isinstance(usage, dict):
                total = usage.get("total_tokens")
                return total if isinstance(total, int) else 0
        usage = getattr(response, "usage", None)
        total = getattr(usage, "total_tokens", None) if usage else None
        return total if isinstance(total, int) else 0

    def _extract_text_content(self, response: Any) -> str:
        if isinstance(response, str):
            return response

        if hasattr(response, "model_dump") and callable(getattr(response, "model_dump")):
            try:
                dumped = response.model_dump()
                if isinstance(dumped, dict):
                    content = self._extract_text_from_dict_payload(dumped)
                    if content:
                        return content
            except Exception:
                pass

        choices = getattr(response, "choices", None)
        if choices:
            normalized_choices: List[Any] = (
                list(choices) if isinstance(choices, list) else [choices]
            )
            first_choice = normalized_choices[0] if normalized_choices else None
        else:
            first_choice = None

        if first_choice is not None:
            message = getattr(first_choice, "message", None)
            if message is None and hasattr(first_choice, "model_dump"):
                try:
                    first_choice_dict = first_choice.model_dump()
                    if isinstance(first_choice_dict, dict):
                        message = first_choice_dict.get("message")
                except Exception:
                    pass
            content = getattr(message, "content", None) if message is not None else None
            if content is None and isinstance(message, dict):
                content = message.get("content")
            extracted = self._extract_text_from_content(content)
            if extracted:
                return extracted

        if isinstance(response, dict):
            extracted = self._extract_text_from_dict_payload(response)
            if extracted:
                return extracted

        raise RuntimeError(
            f"{self.provider_name} returned unsupported response payload: {type(response).__name__}"
        )

    def _extract_text_from_dict_payload(self, payload: Dict[str, Any]) -> str:
        choices_dict = payload.get("choices")
        if isinstance(choices_dict, list) and choices_dict:
            first = choices_dict[0]
            if isinstance(first, dict):
                message_dict = first.get("message")
                if isinstance(message_dict, dict):
                    content = message_dict.get("content")
                    extracted = self._extract_text_from_content(content)
                    if extracted:
                        return extracted
                if "text" in first:
                    extracted = self._extract_text_from_content(first.get("text"))
                    if extracted:
                        return extracted
        extracted = self._extract_text_from_content(payload.get("content"))
        if extracted:
            return extracted
        return ""

    def _extract_text_from_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    text_value = item.get("text")
                    if isinstance(text_value, str) and text_value:
                        parts.append(text_value)
                elif hasattr(item, "text"):
                    text_value = getattr(item, "text", None)
                    if isinstance(text_value, str) and text_value:
                        parts.append(text_value)
            if parts:
                return "\n".join(parts)
        return ""

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
