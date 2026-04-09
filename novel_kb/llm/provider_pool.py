"""多 Provider 轮询池，用于突破单个 Provider 的 QPS 限制"""

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional

from novel_kb.llm.models import AnalysisResult, EmbeddingResult
from novel_kb.llm.provider import LLMProvider
from novel_kb.utils.logger import logger
from novel_kb.utils.errors import ProviderError, ProviderErrorDecision, ProviderFailureAction


class _AsyncRateLimiter:
    def __init__(self, qps: float) -> None:
        self._qps = qps
        self._next_time = 0.0
        self._lock = asyncio.Lock()

    def set_qps(self, qps: float) -> None:
        self._qps = qps

    async def wait(self) -> None:
        if not self._qps or self._qps <= 0:
            return
        import time
        async with self._lock:
            interval = 1.0 / self._qps
            now = time.monotonic()
            if now < self._next_time:
                sleep_for = self._next_time - now
                if sleep_for >= 0.5:
                    logger.info("Rate limiting: sleeping %.2fs", sleep_for)
                await asyncio.sleep(sleep_for)
            self._next_time = max(now, self._next_time) + interval


@dataclass
class _ProviderState:
    provider: LLMProvider
    base_concurrency: int
    base_qps: float
    current_concurrency: int
    current_qps: float
    max_concurrency: int
    max_qps: float
    in_flight: int = 0
    disable_new: bool = False
    success_streak: int = 0
    failure_streak: int = 0
    throttle_events: int = 0
    limiter: Optional[_AsyncRateLimiter] = None
    condition: Optional[asyncio.Condition] = None


class ProviderPool(LLMProvider):
    """Provider 池，轮询多个 provider 以提高并发能力"""
    
    def __init__(self, providers: List[LLMProvider]) -> None:
        if not providers:
            raise ValueError("ProviderPool requires at least one provider")
        self.providers = providers
        self._index = 0
        self._lock = asyncio.Lock()
        self._states: Dict[LLMProvider, _ProviderState] = {}
        for provider in providers:
            base_concurrency = max(1, int(getattr(provider, "concurrency_limit", 1)))
            base_qps = float(getattr(provider, "qps_limit", 0.0) or 0.0)
            limiter = _AsyncRateLimiter(base_qps)
            state = _ProviderState(
                provider=provider,
                base_concurrency=base_concurrency,
                base_qps=base_qps,
                current_concurrency=base_concurrency,
                current_qps=base_qps,
                max_concurrency=max(1, base_concurrency * 2),
                max_qps=base_qps * 2 if base_qps > 0 else 0.0,
                limiter=limiter,
                condition=asyncio.Condition(),
            )
            self._states[provider] = state
        logger.info("ProviderPool initialized with %d providers", len(providers))

    def configure_limits(self, concurrency_limit: int, qps_limit: float) -> None:
        for state in self._states.values():
            base_concurrency = max(1, int(concurrency_limit))
            base_qps = float(qps_limit or 0.0)
            state.base_concurrency = base_concurrency
            state.base_qps = base_qps
            state.max_concurrency = max(1, base_concurrency * 2)
            state.max_qps = base_qps * 2 if base_qps > 0 else 0.0
            state.current_concurrency = base_concurrency
            state.current_qps = base_qps
            if state.limiter:
                state.limiter.set_qps(base_qps)
    
    def _get_next_provider(self) -> LLMProvider:
        """轮询获取下一个 provider（线程安全）"""
        # 简单的轮询策略，不需要锁（Python 中整数赋值是原子的）
        if not self.providers:
            raise ProviderError("No providers available")
        for _ in range(len(self.providers)):
            provider = self.providers[self._index % len(self.providers)]
            self._index += 1
            state = self._states.get(provider)
            if state and state.disable_new:
                continue
            return provider
        raise ProviderError("No providers available (all disabled)")

    async def _acquire_provider(self, provider: LLMProvider) -> None:
        state = self._states.get(provider)
        if not state or not state.condition or not state.limiter:
            return
        async with state.condition:
            while state.disable_new or state.in_flight >= state.current_concurrency:
                await state.condition.wait()
            state.in_flight += 1
        await state.limiter.wait()

    async def _release_provider(self, provider: LLMProvider) -> None:
        state = self._states.get(provider)
        if not state or not state.condition:
            return
        async with state.condition:
            state.in_flight = max(0, state.in_flight - 1)
            state.condition.notify_all()
        if state.disable_new and state.in_flight == 0:
            await self._remove_provider(provider)

    async def _remove_provider(self, provider: LLMProvider) -> None:
        async with self._lock:
            if provider in self.providers:
                desc = self._describe_provider(provider)
                self.providers = [p for p in self.providers if p is not provider]
                self._states.pop(provider, None)
                self._index = 0
                logger.warning("Provider removed from pool: %s", desc)

    def _apply_success(self, provider: LLMProvider) -> None:
        state = self._states.get(provider)
        if not state:
            return
        state.failure_streak = 0
        state.success_streak += 1
        if state.success_streak >= 20:
            if state.current_concurrency < state.max_concurrency:
                state.current_concurrency += 1
            if state.current_qps > 0 and state.current_qps < state.max_qps:
                state.current_qps = min(state.max_qps, state.current_qps * 1.1)
                if state.limiter:
                    state.limiter.set_qps(state.current_qps)
            state.success_streak = 0

    def _apply_failure(self, provider: LLMProvider) -> None:
        state = self._states.get(provider)
        if not state:
            return
        state.success_streak = 0
        state.failure_streak += 1

    def _apply_decision(self, provider: LLMProvider, decision: Optional[ProviderErrorDecision]) -> None:
        if not decision:
            return
        state = self._states.get(provider)
        if not state:
            return
        if decision.action == ProviderFailureAction.DISABLE:
            state.disable_new = True
            logger.warning(
                "Provider disabled: %s reason=%s",
                self._describe_provider(provider),
                decision.reason or "unknown",
            )
            return
        if decision.action == ProviderFailureAction.THROTTLE:
            state.throttle_events += 1
            state.current_concurrency = max(1, state.current_concurrency - 1)
            if state.current_qps > 0:
                state.current_qps = max(0.2, state.current_qps * 0.8)
                if state.limiter:
                    state.limiter.set_qps(state.current_qps)
            
            logger.warning(
                "Provider %s throttled due to %s. New limits -> concurrency: %d, qps: %.2f",
                self._describe_provider(provider),
                decision.reason or "unknown",
                state.current_concurrency,
                state.current_qps,
            )
            
            if state.condition:
                async def _notify() -> None:
                    async with state.condition:
                        state.condition.notify_all()
                asyncio.create_task(_notify())

    @staticmethod
    def _describe_provider(provider: LLMProvider) -> str:
        provider_type = getattr(provider, "provider_type", provider.__class__.__name__)
        model_name = getattr(provider, "model_name", None)
        if model_name:
            return f"{provider_type}:{model_name}"
        return str(provider_type)

    @staticmethod
    def _compact_error_message(exc: Exception, max_chars: int = 260) -> str:
        text = str(exc or "")
        lowered = text.lower()
        if "<!doctype html" in lowered or "<html" in lowered:
            if "502" in lowered or "bad gateway" in lowered:
                return "upstream_502_bad_gateway_html"
            return "upstream_html_error_response"
        compact = " ".join(text.split())
        if len(compact) <= max_chars:
            return compact
        return compact[: max_chars - 1] + "…"

    async def _call_with_context(self, provider: LLMProvider, func_name: str, text: str):
        try:
            await self._acquire_provider(provider)
            func = getattr(provider, func_name)
            result = await func(text)
            # Attach provider metadata if possible
            if hasattr(result, "provider_type"):
                result.provider_type = getattr(provider, "provider_type", None)
                result.model_name = getattr(provider, "model_name", None)
            self._apply_success(provider)
            return result
        except Exception as exc:
            self._apply_failure(provider)
            decision = provider.classify_error(exc)
            self._apply_decision(provider, decision)
            desc = self._describe_provider(provider)
            compact = self._compact_error_message(exc)
            raise ProviderError(f"Provider {desc} failed in {func_name}: {compact}") from exc
        finally:
            await self._release_provider(provider)
    
    async def analyze_characters(self, text: str) -> AnalysisResult:
        provider = self._get_next_provider()
        return await self._call_with_context(provider, "analyze_characters", text)
    
    async def analyze_plot(self, text: str) -> AnalysisResult:
        provider = self._get_next_provider()
        return await self._call_with_context(provider, "analyze_plot", text)
    
    async def extract_features(self, text: str) -> AnalysisResult:
        provider = self._get_next_provider()
        return await self._call_with_context(provider, "extract_features", text)
    
    async def generate_embedding(self, text: str) -> EmbeddingResult:
        provider = self._get_next_provider()
        return await provider.generate_embedding(text)
    
    async def health_check(self) -> bool:
        """检查所有 provider 的健康状态"""
        results = await asyncio.gather(
            *[provider.health_check() for provider in self.providers],
            return_exceptions=True
        )
        healthy_count = sum(1 for r in results if r is True)
        logger.info(
            "ProviderPool health check: %d/%d providers healthy",
            healthy_count,
            len(self.providers)
        )
        return healthy_count > 0  # 至少有一个健康的就返回 True
