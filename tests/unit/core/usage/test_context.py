"""Tests for usage context module."""

import threading
import pytest
from concurrent.futures import ThreadPoolExecutor

from src.core.usage.context import (
    UsageContext,
    usage_context,
    get_current_context,
    set_context,
    clear_context,
)


class TestUsageContext:
    """Tests for UsageContext dataclass."""

    def test_usage_context_basic(self):
        """Test basic UsageContext creation."""
        ctx = UsageContext(
            org_id="org-123",
            feature="document_agent",
        )

        assert ctx.org_id == "org-123"
        assert ctx.feature == "document_agent"
        assert ctx.user_id is None
        assert ctx.session_id is None
        assert ctx.metadata == {}

    def test_usage_context_full(self):
        """Test UsageContext with all fields."""
        ctx = UsageContext(
            org_id="org-123",
            feature="sheets_agent",
            user_id="user-456",
            session_id="session-789",
            request_id="req-abc",
            metadata={"key": "value"},
        )

        assert ctx.org_id == "org-123"
        assert ctx.feature == "sheets_agent"
        assert ctx.user_id == "user-456"
        assert ctx.session_id == "session-789"
        assert ctx.request_id == "req-abc"
        assert ctx.metadata == {"key": "value"}


class TestContextManager:
    """Tests for usage_context context manager."""

    def test_context_manager_sets_and_clears(self):
        """Test that context manager sets and clears context."""
        assert get_current_context() is None

        with usage_context(
            org_id="org-123",
            feature="document_agent",
        ) as ctx:
            assert ctx.org_id == "org-123"
            assert get_current_context() is not None
            assert get_current_context().org_id == "org-123"

        # After exiting context, should be cleared
        assert get_current_context() is None

    def test_context_manager_with_all_params(self):
        """Test context manager with all parameters."""
        with usage_context(
            org_id="org-456",
            feature="sheets_agent",
            user_id="user-abc",
            session_id="session-xyz",
            request_id="req-123",
            metadata={"doc": "test.pdf"},
        ) as ctx:
            assert ctx.org_id == "org-456"
            assert ctx.feature == "sheets_agent"
            assert ctx.user_id == "user-abc"
            assert ctx.session_id == "session-xyz"
            assert ctx.request_id == "req-123"
            assert ctx.metadata == {"doc": "test.pdf"}

    def test_context_manager_exception_handling(self):
        """Test that context is cleared even on exception."""
        try:
            with usage_context(
                org_id="org-123",
                feature="document_agent",
            ):
                assert get_current_context() is not None
                raise ValueError("Test error")
        except ValueError:
            pass

        # Context should be cleared even after exception
        assert get_current_context() is None


class TestThreadLocalContext:
    """Tests for thread-local context isolation."""

    def test_thread_isolation(self):
        """Test that context is isolated per thread."""
        results = {}

        def thread_func(thread_id, org_id):
            with usage_context(
                org_id=org_id,
                feature="test_feature",
            ):
                # Each thread should see its own context
                ctx = get_current_context()
                results[thread_id] = ctx.org_id if ctx else None

        # Run multiple threads with different org_ids
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(thread_func, 1, "org-1"),
                executor.submit(thread_func, 2, "org-2"),
                executor.submit(thread_func, 3, "org-3"),
            ]
            for f in futures:
                f.result()

        # Each thread should have seen its own org_id
        assert results[1] == "org-1"
        assert results[2] == "org-2"
        assert results[3] == "org-3"

    def test_no_cross_thread_contamination(self):
        """Test that one thread's context doesn't affect another."""
        main_context_before = get_current_context()

        def set_context_in_thread():
            with usage_context(
                org_id="thread-org",
                feature="thread_feature",
            ):
                return get_current_context()

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(set_context_in_thread)
            thread_ctx = future.result()

        # Main thread should not be affected
        assert get_current_context() is main_context_before
        # Thread context was set correctly
        assert thread_ctx is not None
        assert thread_ctx.org_id == "thread-org"


class TestSetAndClearContext:
    """Tests for set_context and clear_context functions."""

    def test_set_context(self):
        """Test setting context directly."""
        ctx = UsageContext(
            org_id="org-manual",
            feature="manual_feature",
        )

        set_context(ctx)

        current = get_current_context()
        assert current is not None
        assert current.org_id == "org-manual"

        # Cleanup
        clear_context()

    def test_clear_context(self):
        """Test clearing context."""
        set_context(UsageContext(org_id="org-test", feature="test"))
        assert get_current_context() is not None

        clear_context()

        assert get_current_context() is None

    def test_get_current_context_returns_none_initially(self):
        """Test that get_current_context returns None when not set."""
        clear_context()  # Ensure clean state
        assert get_current_context() is None
