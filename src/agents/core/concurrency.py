"""Async concurrency control for agent invocations.

Provides semaphore-based concurrency limiting as a replacement for
thread pool-based concurrency control. This allows native async
agent execution with `.ainvoke()` while maintaining bounded concurrency.
"""

import asyncio
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Default concurrency limit (matches previous AGENT_EXECUTOR_POOL_SIZE)
DEFAULT_MAX_CONCURRENT = int(os.getenv("AGENT_MAX_CONCURRENT", "10"))


class AgentConcurrency:
    """Shared concurrency control for agent invocations.

    Uses asyncio.Semaphore to limit concurrent agent executions,
    replacing thread pool-based concurrency control.

    Usage:
        async def invoke_agent(query: str):
            async with AgentConcurrency.get_semaphore():
                return await agent.ainvoke({"messages": [query]})

    The semaphore is lazily initialized on first access and reused
    for all subsequent calls within the same event loop.
    """

    _semaphore: Optional[asyncio.Semaphore] = None
    _max_concurrent: int = DEFAULT_MAX_CONCURRENT

    @classmethod
    def get_semaphore(cls, max_concurrent: Optional[int] = None) -> asyncio.Semaphore:
        """Get or create the shared semaphore for agent concurrency control.

        Args:
            max_concurrent: Maximum concurrent agent invocations.
                           Uses AGENT_MAX_CONCURRENT env var or default (10).

        Returns:
            asyncio.Semaphore for limiting concurrent operations.

        Note:
            The semaphore is created lazily and cached. Subsequent calls
            ignore the max_concurrent parameter unless reset() is called.
        """
        if cls._semaphore is None:
            limit = max_concurrent or cls._max_concurrent
            cls._semaphore = asyncio.Semaphore(limit)
            logger.info(f"Created agent concurrency semaphore with limit={limit}")
        return cls._semaphore

    @classmethod
    def reset(cls, max_concurrent: Optional[int] = None) -> None:
        """Reset the semaphore (useful for testing or reconfiguration).

        Args:
            max_concurrent: New maximum concurrent limit.
        """
        if max_concurrent is not None:
            cls._max_concurrent = max_concurrent
        cls._semaphore = None
        logger.debug(f"Reset agent concurrency (new limit={cls._max_concurrent})")

    @classmethod
    async def acquire(cls) -> None:
        """Acquire the semaphore (for manual control)."""
        await cls.get_semaphore().acquire()

    @classmethod
    def release(cls) -> None:
        """Release the semaphore (for manual control)."""
        if cls._semaphore is not None:
            cls._semaphore.release()


__all__ = ["AgentConcurrency", "DEFAULT_MAX_CONCURRENT"]
