"""Tests for usage context module."""

import asyncio
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


class TestAsyncioToThreadContext:
    """Tests for context propagation with asyncio.to_thread.

    These tests verify that contextvars properly propagates context
    when using asyncio.to_thread(), which is the recommended pattern
    for running blocking code in async context (Python 3.9+).
    """

    @pytest.mark.asyncio
    async def test_context_propagates_to_thread(self):
        """Test that context is available in asyncio.to_thread calls.

        asyncio.to_thread() automatically inherits context from the
        current task, which enables token tracking in executor threads.
        """
        def read_context_in_thread():
            ctx = get_current_context()
            return ctx.org_id if ctx else None

        with usage_context(org_id="thread-org", feature="test"):
            result = await asyncio.to_thread(read_context_in_thread)

        # Context should propagate to thread via asyncio.to_thread
        assert result == "thread-org"

    @pytest.mark.asyncio
    async def test_context_cleared_after_to_thread_with_context_exit(self):
        """Test that context is properly cleared when exiting the context manager."""
        def get_ctx():
            return get_current_context()

        with usage_context(org_id="temp-org", feature="test"):
            ctx_in_thread = await asyncio.to_thread(get_ctx)
            assert ctx_in_thread is not None
            assert ctx_in_thread.org_id == "temp-org"

        # After context exit, main thread should have no context
        assert get_current_context() is None

    @pytest.mark.asyncio
    async def test_nested_contexts_in_to_thread(self):
        """Test that nested usage contexts work correctly with asyncio.to_thread."""
        def read_context():
            ctx = get_current_context()
            return ctx.org_id if ctx else None

        captured_values = []

        with usage_context(org_id="outer-org", feature="outer"):
            val1 = await asyncio.to_thread(read_context)
            captured_values.append(val1)

            with usage_context(org_id="inner-org", feature="inner"):
                val2 = await asyncio.to_thread(read_context)
                captured_values.append(val2)

            # After inner context exit, outer should be restored
            val3 = await asyncio.to_thread(read_context)
            captured_values.append(val3)

        assert captured_values == ["outer-org", "inner-org", "outer-org"]


class TestRunInExecutorWithContext:
    """Tests for run_in_executor_with_context helper.

    This helper is used by DocumentAgent to run LLM calls in thread pools
    while preserving the usage context for token tracking.
    """

    @pytest.mark.asyncio
    async def test_context_propagates_with_custom_executor(self):
        """Test that context propagates when using custom executor."""
        from src.utils.async_utils import run_in_executor_with_context

        executor = ThreadPoolExecutor(max_workers=1)

        def read_context():
            ctx = get_current_context()
            return ctx.org_id if ctx else None

        with usage_context(org_id="custom-executor-org", feature="test"):
            result = await run_in_executor_with_context(
                executor,
                read_context
            )

        executor.shutdown(wait=True)
        assert result == "custom-executor-org"

    @pytest.mark.asyncio
    async def test_context_propagates_with_default_executor(self):
        """Test that context propagates when using default executor (None)."""
        from src.utils.async_utils import run_in_executor_with_context

        def read_context():
            ctx = get_current_context()
            return ctx.org_id if ctx else None

        with usage_context(org_id="default-executor-org", feature="test"):
            result = await run_in_executor_with_context(None, read_context)

        assert result == "default-executor-org"

    @pytest.mark.asyncio
    async def test_function_with_args_and_kwargs(self):
        """Test that function arguments are passed correctly."""
        from src.utils.async_utils import run_in_executor_with_context

        def func_with_args(a, b, c=None):
            ctx = get_current_context()
            org = ctx.org_id if ctx else None
            return f"{org}:{a}:{b}:{c}"

        with usage_context(org_id="args-test", feature="test"):
            result = await run_in_executor_with_context(
                None,
                func_with_args,
                "arg1",
                "arg2",
                c="kwarg"
            )

        assert result == "args-test:arg1:arg2:kwarg"
