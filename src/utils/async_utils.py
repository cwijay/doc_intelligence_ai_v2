"""Async execution utilities for sync/async interoperability.

This module provides utilities for running async code from synchronous
contexts, handling edge cases like nested event loops.
"""

import asyncio
import logging
from typing import TypeVar, Coroutine, Any

logger = logging.getLogger(__name__)

T = TypeVar("T")


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run an async coroutine from a sync context.

    Handles various edge cases:
    - No event loop running: Creates a new one
    - Event loop already running: Uses nest_asyncio to allow nested calls
    - RuntimeError: Falls back to asyncio.run()

    Args:
        coro: The coroutine to execute

    Returns:
        The result of the coroutine

    Example:
        >>> async def fetch_data():
        ...     return "data"
        >>> result = run_async(fetch_data())
        >>> print(result)
        'data'

    Note:
        This function uses nest_asyncio when an event loop is already running.
        This is necessary for environments like Jupyter notebooks or when
        LangChain tools are called from within async contexts.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Event loop already running - need nest_asyncio for nested calls
            try:
                import nest_asyncio

                nest_asyncio.apply()
                return loop.run_until_complete(coro)
            except ImportError:
                logger.warning(
                    "nest_asyncio not available - cannot run async in nested context. "
                    "Install with: pip install nest-asyncio"
                )
                raise RuntimeError(
                    "Cannot run async code in nested event loop without nest_asyncio"
                )
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop exists - create one
        return asyncio.run(coro)


async def run_sync_in_executor(func, *args, **kwargs):
    """
    Run a synchronous function in a thread pool executor.

    Args:
        func: The synchronous function to run
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The result of the function

    Example:
        >>> def slow_io_operation(x):
        ...     time.sleep(1)
        ...     return x * 2
        >>> result = await run_sync_in_executor(slow_io_operation, 5)
        >>> print(result)
        10
    """
    import functools

    loop = asyncio.get_running_loop()
    if kwargs:
        func = functools.partial(func, **kwargs)
    return await loop.run_in_executor(None, func, *args)
