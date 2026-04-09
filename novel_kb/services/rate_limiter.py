"""异步速率限制器"""

import asyncio
import time
from typing import Optional


class AsyncRateLimiter:
    """异步 QPS 速率限制"""

    def __init__(self, qps: float) -> None:
        self.qps = qps
        self._next_time = 0.0
        self._lock = asyncio.Lock()

    def set_qps(self, qps: float) -> None:
        self.qps = qps

    async def wait(self) -> None:
        if not self.qps or self.qps <= 0:
            return
        async with self._lock:
            interval = 1.0 / self.qps
            now = time.monotonic()
            if now < self._next_time:
                sleep_for = self._next_time - now
                if sleep_for >= 0.5:
                    import logging
                    logging.getLogger(__name__).info("Rate limiting: sleeping %.2fs", sleep_for)
                await asyncio.sleep(sleep_for)
            self._next_time = max(now, self._next_time) + interval
