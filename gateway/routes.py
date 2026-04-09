import asyncio
import time
from typing import Any, Dict, Optional

from gateway.config import GatewayConfig
from gateway.models import AnalysisResult, EmbeddingResult, GatewayResponse, HealthStatus
from gateway.tier_router import TierRouter


class Routes:
    """HTTP route handlers for the gateway."""

    def __init__(self, config: GatewayConfig) -> None:
        self.config = config
        self.router = TierRouter(config)
        # Global rate limiter (QPS limit per tier)
        self._limiters: Dict[str, asyncio.Semaphore] = {}
        self._rate_limiters: Dict[str, Any] = {}

    async def handle_analyze(self, method: str, tier: str, text: str) -> GatewayResponse:
        """Handle analyze requests (analyze_plot, analyze_characters, extract_features)."""
        start_time = time.monotonic()

        # Get provider for tier
        provider = await self.router.get_provider_for_tier(tier)
        if not provider:
            return GatewayResponse(
                success=False,
                error=f"Invalid tier: {tier}",
                latency_ms=int((time.monotonic() - start_time) * 1000),
            )

        # Apply rate limiting per tier
        await self._get_rate_limiter(tier).wait()

        try:
            if method == "analyze_plot":
                result = await provider.analyze_plot(text)
            elif method == "analyze_characters":
                result = await provider.analyze_characters(text)
            elif method == "extract_features":
                result = await provider.extract_features(text)
            else:
                return GatewayResponse(
                    success=False,
                    error=f"Unknown method: {method}",
                    latency_ms=int((time.monotonic() - start_time) * 1000),
                )

            return GatewayResponse(
                success=True,
                result={
                    "kind": result.kind,
                    "content": result.content,
                    "tokens_used": result.tokens_used,
                    "provider_type": result.provider_type,
                    "model_name": result.model_name,
                },
                latency_ms=int((time.monotonic() - start_time) * 1000),
            )
        except Exception as exc:
            return GatewayResponse(
                success=False,
                error=str(exc),
                latency_ms=int((time.monotonic() - start_time) * 1000),
            )

    async def handle_embed(self, tier: str, text: str) -> GatewayResponse:
        """Handle embedding requests."""
        start_time = time.monotonic()

        provider = await self.router.get_provider_for_tier(tier)
        if not provider:
            return GatewayResponse(
                success=False,
                error=f"Invalid tier: {tier}",
                latency_ms=int((time.monotonic() - start_time) * 1000),
            )

        await self._get_rate_limiter(tier).wait()

        try:
            result = await provider.generate_embedding(text)
            return GatewayResponse(
                success=True,
                result={
                    "vector": result.vector,
                    "tokens_used": result.tokens_used,
                },
                latency_ms=int((time.monotonic() - start_time) * 1000),
            )
        except Exception as exc:
            return GatewayResponse(
                success=False,
                error=str(exc),
                latency_ms=int((time.monotonic() - start_time) * 1000),
            )

    async def handle_health(self) -> GatewayResponse:
        """Handle health check requests."""
        provider_status = await self.router.health_check_all()
        active_tiers = self.router.get_active_tiers()
        healthy = any(provider_status.values())

        return GatewayResponse(
            success=True,
            result={
                "healthy": healthy,
                "providers": provider_status,
                "active_tiers": active_tiers,
            },
            latency_ms=0,
        )

    def _get_rate_limiter(self, tier: str) -> "AsyncRateLimiter":
        """Get or create rate limiter for tier."""
        if tier not in self._rate_limiters:
            # Default: 10 QPS per tier
            from gateway.tier_router import AsyncRateLimiter
            self._rate_limiters[tier] = AsyncRateLimiter(10.0)
        return self._rate_limiters[tier]
