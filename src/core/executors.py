"""Centralized thread pool executor management for different workload types.

This module provides dedicated thread pools for different types of operations:
- Agent pool: For heavy LLM agent invocations (long-running, 5-30s)
- I/O pool: For GCS/file operations (quick, 100-500ms)
- Query pool: For DuckDB/SQL queries (medium, 1-5s)

Separating executors prevents long-running agent operations from starving
I/O operations and provides bounded, tunable concurrency control.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import os
import logging

logger = logging.getLogger(__name__)

# Configurable pool sizes via environment variables
AGENT_POOL_SIZE = int(os.getenv("AGENT_EXECUTOR_POOL_SIZE", "10"))
IO_POOL_SIZE = int(os.getenv("IO_EXECUTOR_POOL_SIZE", "20"))
QUERY_POOL_SIZE = int(os.getenv("QUERY_EXECUTOR_POOL_SIZE", "10"))


class ExecutorRegistry:
    """Manages dedicated thread pools by workload type.

    This class provides separate ThreadPoolExecutor instances for different
    types of operations, allowing fine-grained control over concurrency and
    preventing resource starvation.

    Attributes:
        agent_executor: Pool for LLM agent invocations (LangGraph/LangChain)
        io_executor: Pool for I/O operations (GCS uploads/downloads)
        query_executor: Pool for database/DuckDB queries
    """

    def __init__(self):
        """Initialize all executor pools with configured sizes."""
        self.agent_executor = ThreadPoolExecutor(
            max_workers=AGENT_POOL_SIZE,
            thread_name_prefix="agent-"
        )
        self.io_executor = ThreadPoolExecutor(
            max_workers=IO_POOL_SIZE,
            thread_name_prefix="io-"
        )
        self.query_executor = ThreadPoolExecutor(
            max_workers=QUERY_POOL_SIZE,
            thread_name_prefix="query-"
        )
        logger.info(
            f"ExecutorRegistry initialized: agent={AGENT_POOL_SIZE}, "
            f"io={IO_POOL_SIZE}, query={QUERY_POOL_SIZE} threads"
        )

    def shutdown(self, wait: bool = True, cancel_futures: bool = False):
        """Shutdown all executors gracefully.

        Args:
            wait: If True, wait for all pending futures to complete.
            cancel_futures: If True, cancel pending futures (Python 3.9+).
        """
        logger.info(f"Shutting down ExecutorRegistry (wait={wait})")
        self.agent_executor.shutdown(wait=wait, cancel_futures=cancel_futures)
        self.io_executor.shutdown(wait=wait, cancel_futures=cancel_futures)
        self.query_executor.shutdown(wait=wait, cancel_futures=cancel_futures)
        logger.info("ExecutorRegistry shutdown complete")

    def get_stats(self) -> dict:
        """Return current executor configuration for monitoring.

        Returns:
            Dict with pool configurations and metadata.
        """
        return {
            "agent_pool": {
                "max_workers": AGENT_POOL_SIZE,
                "thread_prefix": "agent-"
            },
            "io_pool": {
                "max_workers": IO_POOL_SIZE,
                "thread_prefix": "io-"
            },
            "query_pool": {
                "max_workers": QUERY_POOL_SIZE,
                "thread_prefix": "query-"
            }
        }


# Module-level singleton instance
_registry: Optional[ExecutorRegistry] = None


def get_executors() -> ExecutorRegistry:
    """Get or create the global executor registry singleton.

    Returns:
        The global ExecutorRegistry instance.
    """
    global _registry
    if _registry is None:
        _registry = ExecutorRegistry()
    return _registry


def shutdown_executors(wait: bool = True):
    """Shutdown the global executor registry if initialized.

    This should be called during application shutdown to cleanly
    terminate all thread pools.

    Args:
        wait: If True, wait for all pending futures to complete.
    """
    global _registry
    if _registry is not None:
        _registry.shutdown(wait=wait)
        _registry = None
