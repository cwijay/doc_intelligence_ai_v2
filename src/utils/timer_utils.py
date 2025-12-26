"""Timing and performance measurement utilities.

This module provides utilities for measuring elapsed time in a consistent
manner throughout the codebase.
"""

import time
from typing import Optional


def elapsed_ms(start_time: float) -> float:
    """
    Calculate elapsed time in milliseconds since start_time.

    Args:
        start_time: Start time from time.time()

    Returns:
        Elapsed time in milliseconds

    Example:
        >>> start = time.time()
        >>> # ... do some work ...
        >>> duration = elapsed_ms(start)
        >>> print(f"Took {duration:.2f}ms")
    """
    return (time.time() - start_time) * 1000


class Timer:
    """
    Context manager for timing code blocks.

    Usage:
        >>> with Timer() as t:
        ...     # do some work
        ...     pass
        >>> print(f"Elapsed: {t.elapsed_ms:.2f}ms")

    Can also be used as a callable:
        >>> timer = Timer()
        >>> timer.start()
        >>> # do some work
        >>> print(f"Elapsed: {timer.elapsed_ms:.2f}ms")
    """

    def __init__(self):
        self.start_time: Optional[float] = None
        self._elapsed_ms: float = 0

    def start(self) -> "Timer":
        """Start the timer."""
        self.start_time = time.time()
        self._elapsed_ms = 0
        return self

    def stop(self) -> float:
        """Stop the timer and return elapsed milliseconds."""
        if self.start_time is not None:
            self._elapsed_ms = elapsed_ms(self.start_time)
        return self._elapsed_ms

    @property
    def elapsed_ms(self) -> float:
        """
        Get elapsed time in milliseconds.

        If timer is still running, returns current elapsed time.
        If timer is stopped, returns final elapsed time.
        """
        if self.start_time is None:
            return self._elapsed_ms
        if self._elapsed_ms > 0:
            return self._elapsed_ms
        return elapsed_ms(self.start_time)

    def __enter__(self) -> "Timer":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()

    def __repr__(self) -> str:
        return f"Timer(elapsed_ms={self.elapsed_ms:.2f})"
