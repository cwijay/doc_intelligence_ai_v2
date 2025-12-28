"""Unit tests for RateLimiter class."""

import pytest
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.agents.core.rate_limiter import RateLimiter


class TestRateLimiterInit:
    """Tests for RateLimiter initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        limiter = RateLimiter()
        assert limiter.max_requests == 10
        assert limiter.window_seconds == 60

    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        limiter = RateLimiter(max_requests=5, window_seconds=30)
        assert limiter.max_requests == 5
        assert limiter.window_seconds == 30

    def test_repr(self):
        """Test string representation."""
        limiter = RateLimiter(max_requests=5, window_seconds=30)
        repr_str = repr(limiter)
        assert "max_requests=5" in repr_str
        assert "window_seconds=30" in repr_str
        assert "tracked_sessions=" in repr_str

    def test_requests_dict_initialized(self):
        """Test requests dictionary is initialized."""
        limiter = RateLimiter()
        assert isinstance(limiter.requests, dict)
        assert len(limiter.requests) == 0


class TestIsAllowed:
    """Tests for is_allowed method."""

    def test_first_request_allowed(self):
        """Test that first request is always allowed."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        assert limiter.is_allowed("session-1") is True

    def test_within_limit_allowed(self):
        """Test requests within limit are allowed."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            assert limiter.is_allowed("session-1") is True

    def test_at_limit_denied(self):
        """Test request at limit is denied."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            limiter.is_allowed("session-1")
        assert limiter.is_allowed("session-1") is False

    def test_over_limit_denied(self):
        """Test requests over limit continue to be denied."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        limiter.is_allowed("session-1")
        limiter.is_allowed("session-1")
        assert limiter.is_allowed("session-1") is False
        assert limiter.is_allowed("session-1") is False

    def test_different_sessions_independent(self):
        """Test that different sessions have independent limits."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        assert limiter.is_allowed("session-1") is True
        assert limiter.is_allowed("session-1") is True
        assert limiter.is_allowed("session-1") is False  # Session 1 at limit
        assert limiter.is_allowed("session-2") is True  # Session 2 still allowed
        assert limiter.is_allowed("session-2") is True

    def test_window_expiration(self):
        """Test that requests outside window are not counted."""
        limiter = RateLimiter(max_requests=2, window_seconds=1)
        limiter.is_allowed("session-1")
        limiter.is_allowed("session-1")
        assert limiter.is_allowed("session-1") is False

        time.sleep(1.1)  # Wait for window to expire
        assert limiter.is_allowed("session-1") is True

    def test_partial_window_expiration(self):
        """Test partial window expiration allows new requests."""
        limiter = RateLimiter(max_requests=2, window_seconds=1)
        limiter.is_allowed("session-1")
        time.sleep(0.6)
        limiter.is_allowed("session-1")
        # First request should be expired now
        time.sleep(0.5)
        assert limiter.is_allowed("session-1") is True

    def test_empty_session_id(self):
        """Test handling of empty session ID."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        assert limiter.is_allowed("") is True
        assert limiter.is_allowed("") is True

    def test_is_allowed_tracks_request(self):
        """Test that is_allowed automatically tracks requests."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        limiter.is_allowed("session-1")
        limiter.is_allowed("session-1")

        assert len(limiter.requests["session-1"]) == 2


class TestGetRetryAfter:
    """Tests for get_retry_after method."""

    def test_no_requests_returns_zero(self):
        """Test retry_after is 0 when no requests made."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        assert limiter.get_retry_after("session-1") == 0

    def test_unknown_session_returns_zero(self):
        """Test retry_after is 0 for unknown session."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        limiter.is_allowed("session-1")
        assert limiter.get_retry_after("session-unknown") == 0

    def test_at_limit_returns_positive(self):
        """Test retry_after is positive when at limit."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        limiter.is_allowed("session-1")
        limiter.is_allowed("session-1")
        retry_after = limiter.get_retry_after("session-1")
        assert 0 < retry_after <= 60

    def test_retry_after_decreases_over_time(self):
        """Test that retry_after decreases as time passes."""
        limiter = RateLimiter(max_requests=1, window_seconds=10)
        limiter.is_allowed("session-1")

        retry1 = limiter.get_retry_after("session-1")
        time.sleep(2)
        retry2 = limiter.get_retry_after("session-1")

        assert retry2 < retry1

    def test_retry_after_never_negative(self):
        """Test retry_after is never negative."""
        limiter = RateLimiter(max_requests=1, window_seconds=1)
        limiter.is_allowed("session-1")

        time.sleep(1.5)
        retry = limiter.get_retry_after("session-1")
        assert retry >= 0


class TestGetRemaining:
    """Tests for get_remaining method."""

    def test_no_requests_returns_max(self):
        """Test remaining equals max when no requests made."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        assert limiter.get_remaining("session-1") == 5

    def test_unknown_session_returns_max(self):
        """Test remaining equals max for unknown session."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        limiter.is_allowed("session-1")
        assert limiter.get_remaining("session-unknown") == 5

    def test_after_requests_decreases(self):
        """Test remaining decreases after requests."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        limiter.is_allowed("session-1")
        limiter.is_allowed("session-1")
        assert limiter.get_remaining("session-1") == 3

    def test_at_limit_returns_zero(self):
        """Test remaining is 0 when at limit."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        limiter.is_allowed("session-1")
        limiter.is_allowed("session-1")
        assert limiter.get_remaining("session-1") == 0

    def test_never_negative(self):
        """Test remaining never goes negative."""
        limiter = RateLimiter(max_requests=1, window_seconds=60)
        limiter.is_allowed("session-1")
        # Try more requests (will be denied but still attempt)
        limiter.is_allowed("session-1")
        limiter.is_allowed("session-1")
        assert limiter.get_remaining("session-1") >= 0

    def test_remaining_increases_after_window_expires(self):
        """Test remaining increases after requests expire."""
        limiter = RateLimiter(max_requests=2, window_seconds=1)
        limiter.is_allowed("session-1")
        limiter.is_allowed("session-1")
        assert limiter.get_remaining("session-1") == 0

        time.sleep(1.1)
        assert limiter.get_remaining("session-1") == 2


class TestCleanup:
    """Tests for cleanup method."""

    def test_cleanup_empty(self):
        """Test cleanup on empty limiter."""
        limiter = RateLimiter()
        assert limiter.cleanup() == 0

    def test_cleanup_stale_sessions(self):
        """Test cleanup removes stale sessions."""
        limiter = RateLimiter(max_requests=5, window_seconds=1)
        limiter.is_allowed("session-1")
        limiter.is_allowed("session-2")

        time.sleep(1.1)

        cleaned = limiter.cleanup()
        assert cleaned == 2
        assert "session-1" not in limiter.requests
        assert "session-2" not in limiter.requests

    def test_cleanup_preserves_active_sessions(self):
        """Test cleanup preserves active sessions."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        limiter.is_allowed("session-1")

        cleaned = limiter.cleanup()
        assert cleaned == 0
        assert "session-1" in limiter.requests

    def test_cleanup_partial(self):
        """Test cleanup removes only stale sessions."""
        limiter = RateLimiter(max_requests=5, window_seconds=1)
        limiter.is_allowed("session-stale")

        time.sleep(1.1)

        limiter.is_allowed("session-active")
        cleaned = limiter.cleanup()

        assert cleaned == 1
        assert "session-stale" not in limiter.requests
        assert "session-active" in limiter.requests


class TestReset:
    """Tests for reset method."""

    def test_reset_clears_session(self):
        """Test reset clears session data."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        limiter.is_allowed("session-1")
        limiter.is_allowed("session-1")
        assert limiter.is_allowed("session-1") is False

        limiter.reset("session-1")
        assert limiter.is_allowed("session-1") is True

    def test_reset_unknown_session_no_error(self):
        """Test reset on unknown session doesn't error."""
        limiter = RateLimiter()
        limiter.reset("unknown-session")  # Should not raise

    def test_reset_one_session_preserves_others(self):
        """Test reset only affects specified session."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        limiter.is_allowed("session-1")
        limiter.is_allowed("session-1")
        limiter.is_allowed("session-2")
        limiter.is_allowed("session-2")

        limiter.reset("session-1")

        assert limiter.get_remaining("session-1") == 2
        assert limiter.get_remaining("session-2") == 0


class TestThreadSafety:
    """Tests for thread-safety of RateLimiter."""

    def test_concurrent_requests_same_session(self):
        """Test concurrent requests to same session."""
        limiter = RateLimiter(max_requests=100, window_seconds=60)
        results = []

        def make_request():
            return limiter.is_allowed("session-1")

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(100)]
            results = [f.result() for f in as_completed(futures)]

        # All 100 should be allowed
        assert sum(results) == 100
        assert limiter.get_remaining("session-1") == 0

    def test_concurrent_requests_exceed_limit(self):
        """Test concurrent requests that exceed limit."""
        limiter = RateLimiter(max_requests=50, window_seconds=60)
        results = []

        def make_request():
            return limiter.is_allowed("session-1")

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(make_request) for _ in range(100)]
            results = [f.result() for f in as_completed(futures)]

        # Only 50 should be allowed
        assert sum(results) == 50
        assert results.count(False) == 50

    def test_concurrent_different_sessions(self):
        """Test concurrent requests to different sessions."""
        limiter = RateLimiter(max_requests=10, window_seconds=60)

        def make_requests_for_session(session_id):
            allowed = 0
            for _ in range(15):
                if limiter.is_allowed(session_id):
                    allowed += 1
            return allowed

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(make_requests_for_session, f"session-{i}")
                for i in range(5)
            ]
            results = [f.result() for f in as_completed(futures)]

        # Each session should have exactly 10 allowed
        assert all(r == 10 for r in results)

    def test_concurrent_cleanup_and_requests(self):
        """Test cleanup running concurrently with requests."""
        limiter = RateLimiter(max_requests=1000, window_seconds=0.1)
        errors = []

        def make_requests():
            try:
                for _ in range(100):
                    limiter.is_allowed("session-1")
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)

        def run_cleanup():
            try:
                for _ in range(10):
                    limiter.cleanup()
                    time.sleep(0.05)
            except Exception as e:
                errors.append(e)

        # Should not deadlock or raise
        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(make_requests)
            executor.submit(run_cleanup)

        assert len(errors) == 0

    def test_concurrent_reset_and_requests(self):
        """Test reset running concurrently with requests."""
        limiter = RateLimiter(max_requests=100, window_seconds=60)
        errors = []

        def make_requests():
            try:
                for _ in range(100):
                    limiter.is_allowed("session-1")
            except Exception as e:
                errors.append(e)

        def run_resets():
            try:
                for _ in range(10):
                    limiter.reset("session-1")
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(make_requests)
            executor.submit(run_resets)

        assert len(errors) == 0
