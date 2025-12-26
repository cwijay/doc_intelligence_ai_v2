"""Thread-safe rate limiter for controlling request frequency per session.

This module provides a shared RateLimiter class that can be used by both
SheetsAgent and DocumentAgent to enforce rate limits on API requests.
"""

import logging
import threading
import time
from collections import defaultdict
from typing import Dict, List

from ...constants import DEFAULT_RATE_LIMIT_REQUESTS, DEFAULT_RATE_LIMIT_WINDOW_SECONDS

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Thread-safe rate limiter for controlling request frequency per session.

    Implements a sliding window rate limiting algorithm where requests are
    tracked per session and limited to a configurable maximum within a
    time window.

    Attributes:
        max_requests: Maximum number of requests allowed per window
        window_seconds: Time window in seconds for rate limiting

    Example:
        >>> limiter = RateLimiter(max_requests=10, window_seconds=60)
        >>> if limiter.is_allowed("session-123"):
        ...     # Process request
        ...     pass
        >>> else:
        ...     retry_after = limiter.get_retry_after("session-123")
        ...     print(f"Rate limited. Try again in {retry_after}s")
    """

    def __init__(
        self,
        max_requests: int = DEFAULT_RATE_LIMIT_REQUESTS,
        window_seconds: int = DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
    ):
        """
        Initialize the rate limiter.

        Args:
            max_requests: Maximum requests per window (default: 10)
            window_seconds: Window size in seconds (default: 60)
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def is_allowed(self, session_id: str) -> bool:
        """
        Check if a request is allowed for the given session.

        Automatically tracks the request if allowed.

        Args:
            session_id: Unique session identifier

        Returns:
            True if request is allowed, False if rate limited
        """
        with self._lock:
            now = time.time()
            window_start = now - self.window_seconds

            # Remove expired requests outside the window
            self.requests[session_id] = [
                t for t in self.requests[session_id] if t > window_start
            ]

            # Check if limit exceeded
            if len(self.requests[session_id]) >= self.max_requests:
                logger.warning(f"Rate limit exceeded for session {session_id}")
                return False

            # Track this request
            self.requests[session_id].append(now)
            return True

    def get_retry_after(self, session_id: str) -> int:
        """
        Get seconds until rate limit resets for a session.

        Args:
            session_id: Unique session identifier

        Returns:
            Seconds until the oldest request expires from the window
        """
        with self._lock:
            if session_id not in self.requests or not self.requests[session_id]:
                return 0

            oldest_request = min(self.requests[session_id])
            retry_after = int(oldest_request + self.window_seconds - time.time())
            return max(0, retry_after)

    def get_remaining(self, session_id: str) -> int:
        """
        Get the number of remaining requests allowed for a session.

        Args:
            session_id: Unique session identifier

        Returns:
            Number of requests remaining in the current window
        """
        with self._lock:
            now = time.time()
            window_start = now - self.window_seconds

            # Count requests in current window
            current_requests = len([
                t for t in self.requests.get(session_id, []) if t > window_start
            ])

            return max(0, self.max_requests - current_requests)

    def cleanup(self) -> int:
        """
        Remove stale entries from the rate limiter.

        Returns:
            Number of sessions cleaned up
        """
        with self._lock:
            now = time.time()
            window_start = now - self.window_seconds

            stale_sessions = [
                session_id
                for session_id, requests in self.requests.items()
                if not requests or max(requests) < window_start
            ]

            for session_id in stale_sessions:
                del self.requests[session_id]

            if stale_sessions:
                logger.debug(f"Cleaned up {len(stale_sessions)} stale rate limit entries")

            return len(stale_sessions)

    def reset(self, session_id: str) -> None:
        """
        Reset rate limit for a specific session.

        Args:
            session_id: Session to reset
        """
        with self._lock:
            if session_id in self.requests:
                del self.requests[session_id]

    def __repr__(self) -> str:
        return (
            f"RateLimiter(max_requests={self.max_requests}, "
            f"window_seconds={self.window_seconds}, "
            f"tracked_sessions={len(self.requests)})"
        )
