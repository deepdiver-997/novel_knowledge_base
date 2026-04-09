import asyncio
import time
from typing import Any, Dict, List, Optional

from gateway.config import GatewayConfig, ProviderConfig, TierConfig
from gateway.models import AnalysisResult, EmbeddingResult
from gateway.providers.aliyun import AliyunProvider
from gateway.providers.base import LLMProvider
from gateway.providers.openai_like import OpenAILikeProvider
from gateway.providers.volcengine import VolcEngineProvider
from gateway.providers.xunfei import XunfeiProvider


class TierRouter:
    """Routes tier names to provider instances."""

    def __init__(self, config: GatewayConfig) -> None:
        self.config = config
        self._providers: Dict[str, LLMProvider] = {}
        self._lock = asyncio.Lock()

    async def get_provider_for_tier(self, tier: str) -> Optional[LLMProvider]:
        """Get or create a provider for the given tier."""
        tier_cfg = self.config.tiers.get(tier)
        if not tier_cfg:
            return None

        # Check cache first
        key = f"{tier_cfg.provider}:{tier_cfg.model}"
        if key in self._providers:
            return self._providers[key]

        async with self._lock:
            # Double-check after acquiring lock
            if key in self._providers:
                return self._providers[key]

            # Create provider
            provider = await self._create_provider(tier_cfg.provider, tier_cfg.model)
            if provider:
                self._providers[key] = provider
            return provider

    async def _create_provider(self, provider_name: str, model: str) -> Optional[LLMProvider]:
        """Create a provider instance based on name and model."""
        cfg = self.config.providers.get(provider_name)
        if not cfg:
            return None

        try:
            if provider_name == "aliyun":
                return AliyunProvider(
                    api_key=cfg.api_key,
                    base_url=cfg.base_url,
                    model=model,
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                    timeout=cfg.timeout,
                )
            elif provider_name == "xunfei":
                return XunfeiProvider(
                    api_key=cfg.api_key,
                    base_url=cfg.base_url,
                    model=model,
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                    timeout=cfg.timeout,
                    embedding_model=cfg.embedding_model,
                    lora_id=cfg.lora_id,
                )
            elif provider_name == "volcengine":
                return VolcEngineProvider(
                    api_key=cfg.api_key,
                    base_url=cfg.base_url,
                    model=model,
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                    timeout=cfg.timeout,
                )
            else:
                # Generic OpenAI-compatible provider
                return OpenAILikeProvider(
                    provider_name=provider_name,
                    model=model,
                    api_key=cfg.api_key,
                    base_url=cfg.base_url,
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                    timeout=cfg.timeout,
                    embedding_model=cfg.embedding_model,
                    extra_headers=cfg.extra_headers,
                    extra_body=cfg.extra_body,
                )
        except Exception:
            return None

    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all configured providers."""
        results: Dict[str, bool] = {}
        for name, cfg in self.config.providers.items():
            try:
                provider = await self._create_provider(name, cfg.model)
                if provider:
                    results[name] = await provider.health_check()
                else:
                    results[name] = False
            except Exception:
                results[name] = False
        return results

    def get_active_tiers(self) -> List[str]:
        """Return list of tiers that have valid providers."""
        return list(self.config.tiers.keys())


class AsyncRateLimiter:
    """Simple async rate limiter."""

    def __init__(self, qps: float) -> None:
        self.qps = qps
        self._next_time = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        if not self.qps or self.qps <= 0:
            return
        async with self._lock:
            interval = 1.0 / self.qps
            now = time.monotonic()
            if now < self._next_time:
                await asyncio.sleep(self._next_time - now)
            self._next_time = now + interval
