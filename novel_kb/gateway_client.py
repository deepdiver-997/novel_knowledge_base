"""Gateway client for novel_kb - communicates with LLM Gateway service."""

import asyncio
import http.client
import json
from typing import Any, Dict, List, Optional

from novel_kb.llm.models import AnalysisResult, EmbeddingResult
from novel_kb.llm.provider import LLMProvider


class GatewayClient(LLMProvider):
    """LLM Gateway client - communicates with local gateway service."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8747",
        tier: str = "medium",
        timeout: int = 120,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        # 按文本长度自动选择 tier
        tier_short: Optional[str] = None,  # 短文本用此 tier
        tier_long: Optional[str] = None,  # 长文本用此 tier
        short_text_chars: int = 1000,    # 短文本阈值
        long_text_chars: int = 4000,     # 长文本阈值
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.tier = tier
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        # 自动 tier 选择
        self.tier_short = tier_short
        self.tier_long = tier_long
        self.short_text_chars = short_text_chars
        self.long_text_chars = long_text_chars

    def _tier_for_text(self, text: str) -> str:
        """根据文本长度选择合适的 tier."""
        if not self.tier_short and not self.tier_long:
            return self.tier
        chars = len(text)
        if chars <= self.short_text_chars and self.tier_short:
            return self.tier_short
        if chars >= self.long_text_chars and self.tier_long:
            return self.tier_long
        return self.tier

    def _build_analyze_payload(self, method: str, text: str, tier_override: Optional[str] = None) -> Dict[str, Any]:
        """Build request payload with optional max_tokens/temperature overrides."""
        tier = tier_override if tier_override else self._tier_for_text(text)
        payload = {"method": method, "tier": tier, "text": text}
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        return payload

    def _request_sync(self, method: str, path: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Synchronous HTTP request to gateway using http.client."""
        host = self.base_url.replace("http://", "").split(":")[0]
        port = int(self.base_url.split(":")[-1]) if ":" in self.base_url else 80

        try:
            conn = http.client.HTTPConnection(host, port, timeout=self.timeout)
            body_bytes = json.dumps(data, ensure_ascii=False).encode("utf-8") if data else b""
            headers = {"Content-Type": "application/json; charset=utf-8", "Content-Length": str(len(body_bytes))}

            conn.request(method, path, body=body_bytes, headers=headers)
            response = conn.getresponse()

            response_body = response.read().decode("utf-8")
            conn.close()

            if response.status not in (200, 500):
                return {"success": False, "error": f"HTTP {response.status}"}

            return json.loads(response_body) if response_body else {}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _request(self, method: str, path: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Async HTTP request to gateway using asyncio streams."""
        host = self.base_url.replace("http://", "").split(":")[0]
        port = int(self.base_url.split(":")[-1]) if ":" in self.base_url else 80

        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=self.timeout,
        )

        try:
            body = json.dumps(data, ensure_ascii=False) if data else ""
            body_bytes = body.encode("utf-8")

            request = (
                f"{method} {path} HTTP/1.1\r\n"
                f"Host: {host}:{port}\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(body_bytes)}\r\n"
                f"Connection: close\r\n"
                f"\r\n"
            )
            writer.write(request.encode("utf-8"))
            writer.write(body_bytes)
            await writer.drain()

            # Read response
            response_bytes = await reader.read()
            response = response_bytes.decode("utf-8")

            # Parse HTTP response
            if "\r\n\r\n" not in response:
                return {"success": False, "error": "Invalid response"}

            headers, body = response.split("\r\n\r\n", 1)
            status_line = headers.split("\r\n")[0]

            if " 200 " not in status_line and " 500 " not in status_line:
                return {"success": False, "error": f"HTTP error: {status_line}"}

            return json.loads(body) if body else {}
        finally:
            writer.close()
            await writer.wait_closed()

    async def analyze_plot(self, text: str, tier: Optional[str] = None) -> AnalysisResult:
        normalized_text = str(text or "").strip()
        if not normalized_text:
            raise ValueError("GatewayClient.analyze_plot called with empty text")
        response = await self._request(
            "POST",
            "/v1/analyze",
            self._build_analyze_payload("analyze_plot", normalized_text, tier_override=tier),
        )
        return self._parse_result(response, "plot")

    async def analyze_characters(self, text: str, tier: Optional[str] = None) -> AnalysisResult:
        normalized_text = str(text or "").strip()
        if not normalized_text:
            raise ValueError("GatewayClient.analyze_characters called with empty text")
        response = await self._request(
            "POST",
            "/v1/analyze",
            self._build_analyze_payload("analyze_characters", normalized_text, tier_override=tier),
        )
        return self._parse_result(response, "characters")

    async def extract_features(self, text: str, tier: Optional[str] = None) -> AnalysisResult:
        normalized_text = str(text or "").strip()
        if not normalized_text:
            raise ValueError("GatewayClient.extract_features called with empty text")
        response = await self._request(
            "POST",
            "/v1/analyze",
            self._build_analyze_payload("extract_features", normalized_text, tier_override=tier),
        )
        return self._parse_result(response, "features")

    async def generate_embedding(self, text: str) -> EmbeddingResult:
        response = await self._request(
            "POST",
            "/v1/embed",
            {"tier": self.tier, "text": text},
        )
        result = response.get("result")
        if not result:
            return EmbeddingResult(vector=[], tokens_used=0)
        return EmbeddingResult(
            vector=result.get("vector", []),
            tokens_used=result.get("tokens_used", 0),
        )

    def generate_embedding_sync(self, text: str) -> EmbeddingResult:
        """Synchronous embedding - for use from sync contexts."""
        response = self._request_sync(
            "POST",
            "/v1/embed",
            {"tier": self.tier, "text": text},
        )
        result = response.get("result")
        if not result:
            return EmbeddingResult(vector=[], tokens_used=0)
        return EmbeddingResult(
            vector=result.get("vector", []),
            tokens_used=result.get("tokens_used", 0),
        )

    async def health_check(self) -> bool:
        response = await self._request("GET", "/v1/health")
        return response.get("success", False) and response.get("result", {}).get("healthy", False)

    def _parse_result(self, response: Dict[str, Any], kind: str) -> AnalysisResult:
        """Parse gateway response into AnalysisResult."""
        if not response.get("success"):
            raise RuntimeError(f"Gateway error: {response.get('error', 'Unknown error')}")

        result = response.get("result", {})
        return AnalysisResult(
            kind=kind,
            content=result.get("content", {}),
            tokens_used=result.get("tokens_used", 0),
            provider_type=result.get("provider_type"),
            model_name=result.get("model_name"),
        )
