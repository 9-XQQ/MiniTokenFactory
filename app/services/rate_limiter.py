
import asyncio
import time

from collections.abc import Callable


class TokenBucket:
    """
    进程内令牌桶限流器。

    rate: 每秒补充的令牌数量
    capacity: 桶中最多保存的令牌数量
    """

    def __init__(
        self,
        *,
        rate: float,
        capacity: int,
        clock: Callable[[], float] = time.monotonic,
    ):
        if rate <= 0:
            raise ValueError("rate must be positive")

        if capacity < 1:
            raise ValueError("capacity must be >= 1")

        self._rate = rate
        self._capacity = capacity
        self._tokens = float(capacity)

        self._clock = clock
        self._updated_at = clock()

        self._lock = asyncio.Lock()

    async def allow(self) -> bool:
        """
        尝试消耗一个令牌。

        返回：
            True：请求允许通过
            False：令牌不足，请求被拒绝
        """

        async with self._lock:
            now = self._clock()

            elapsed = max(
                0.0,
                now - self._updated_at,
            )

            # 根据经过的时间补充令牌
            self._tokens = min(
                float(self._capacity),
                self._tokens + elapsed * self._rate,
            )

            self._updated_at = now

            # 令牌不足
            if self._tokens < 1.0:
                return False

            # 消耗一个令牌
            self._tokens -= 1.0

            return True
