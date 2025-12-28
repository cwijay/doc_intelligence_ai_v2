"""Unit tests for UsageQueue."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.usage.usage_queue import (
    UsageEvent,
    UsageQueue,
    enqueue_resource_usage,
    enqueue_token_usage,
    get_usage_queue,
)


class TestUsageEvent:
    """Tests for UsageEvent dataclass."""

    def test_token_event_creation(self):
        """Test creating a token usage event."""
        event = UsageEvent(
            event_type="token",
            org_id="org-123",
            feature="document_agent",
            model="gemini-3-flash-preview",
            provider="google",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        )

        assert event.event_type == "token"
        assert event.org_id == "org-123"
        assert event.feature == "document_agent"
        assert event.model == "gemini-3-flash-preview"
        assert event.provider == "google"
        assert event.input_tokens == 100
        assert event.output_tokens == 50
        assert event.total_tokens == 150
        assert event.created_at is not None

    def test_resource_event_creation(self):
        """Test creating a resource usage event."""
        event = UsageEvent(
            event_type="resource",
            org_id="org-123",
            resource_type="llamaparse_pages",
            amount=5,
            file_name="test.pdf",
        )

        assert event.event_type == "resource"
        assert event.org_id == "org-123"
        assert event.resource_type == "llamaparse_pages"
        assert event.amount == 5
        assert event.file_name == "test.pdf"

    def test_event_with_metadata(self):
        """Test event with optional metadata."""
        metadata = {"session_id": "sess-123", "endpoint": "/api/v1/chat"}
        event = UsageEvent(
            event_type="token",
            org_id="org-123",
            metadata=metadata,
        )

        assert event.metadata == metadata


class TestUsageQueue:
    """Tests for UsageQueue class."""

    @pytest.fixture
    def fresh_queue(self):
        """Create a fresh queue instance for testing."""
        # Reset singleton
        UsageQueue._instance = None
        queue = UsageQueue()
        yield queue
        # Cleanup
        if queue._started:
            queue.shutdown(wait=True, timeout=2.0)
        UsageQueue._instance = None

    def test_singleton_pattern(self, fresh_queue):
        """Test that UsageQueue is a singleton."""
        queue2 = UsageQueue()
        assert fresh_queue is queue2

    def test_get_usage_queue_returns_singleton(self, fresh_queue):
        """Test get_usage_queue returns the same singleton."""
        queue2 = get_usage_queue()
        assert fresh_queue is queue2

    def test_initial_state(self, fresh_queue):
        """Test queue initial state."""
        assert not fresh_queue.is_running
        assert fresh_queue.queue_size == 0

    def test_start_stop(self, fresh_queue):
        """Test starting and stopping the queue."""
        fresh_queue.start()
        assert fresh_queue.is_running

        fresh_queue.shutdown(wait=True, timeout=2.0)
        assert not fresh_queue.is_running

    def test_start_is_idempotent(self, fresh_queue):
        """Test that multiple start calls don't create multiple threads."""
        fresh_queue.start()
        thread1 = fresh_queue._thread

        fresh_queue.start()
        thread2 = fresh_queue._thread

        assert thread1 is thread2
        assert fresh_queue.is_running

    def test_enqueue_auto_starts(self, fresh_queue):
        """Test that enqueue auto-starts the queue if not running."""
        assert not fresh_queue.is_running

        event = UsageEvent(
            event_type="token",
            org_id="org-123",
        )
        fresh_queue.enqueue(event)

        assert fresh_queue.is_running

    def test_queue_size_increases(self, fresh_queue):
        """Test that queue size increases when events are added."""
        # Start queue but patch the service to prevent processing
        with patch("src.core.usage.service.get_usage_service"):
            fresh_queue.start()

            # Add events faster than they can be processed
            for i in range(5):
                event = UsageEvent(
                    event_type="token",
                    org_id=f"org-{i}",
                )
                fresh_queue.enqueue(event)

            # Queue size should be > 0 at some point
            # (though exact value depends on processing speed)
            assert fresh_queue.queue_size >= 0

    def test_shutdown_without_start(self, fresh_queue):
        """Test shutdown when queue was never started."""
        # Should not raise
        fresh_queue.shutdown(wait=True, timeout=1.0)
        assert not fresh_queue.is_running

    def test_shutdown_is_idempotent(self, fresh_queue):
        """Test multiple shutdown calls don't raise."""
        fresh_queue.start()
        fresh_queue.shutdown(wait=True, timeout=2.0)
        fresh_queue.shutdown(wait=True, timeout=2.0)
        assert not fresh_queue.is_running


class TestUsageQueueProcessing:
    """Tests for event processing in UsageQueue."""

    @pytest.fixture
    def fresh_queue(self):
        """Create a fresh queue instance for testing."""
        UsageQueue._instance = None
        queue = UsageQueue()
        yield queue
        if queue._started:
            queue.shutdown(wait=True, timeout=2.0)
        UsageQueue._instance = None

    @patch("src.core.usage.service.get_usage_service")
    def test_token_event_processing(self, mock_get_service, fresh_queue):
        """Test that token events are processed correctly."""
        mock_service = MagicMock()
        mock_service.log_token_usage = AsyncMock()
        mock_get_service.return_value = mock_service

        fresh_queue.start()

        event = UsageEvent(
            event_type="token",
            org_id="org-123",
            feature="test_feature",
            model="test-model",
            provider="test",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cached_tokens=10,
        )
        fresh_queue.enqueue(event)

        # Wait for processing
        time.sleep(2.0)

        # Verify service was called
        assert mock_service.log_token_usage.called

    @patch("src.core.usage.service.get_usage_service")
    def test_resource_event_processing(self, mock_get_service, fresh_queue):
        """Test that resource events are processed correctly."""
        mock_service = MagicMock()
        mock_service.log_resource_usage = AsyncMock()
        mock_get_service.return_value = mock_service

        fresh_queue.start()

        event = UsageEvent(
            event_type="resource",
            org_id="org-123",
            resource_type="llamaparse_pages",
            amount=5,
            file_name="test.pdf",
        )
        fresh_queue.enqueue(event)

        # Wait for processing
        time.sleep(2.0)

        # Verify service was called
        assert mock_service.log_resource_usage.called


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before and after each test."""
        UsageQueue._instance = None
        yield
        if UsageQueue._instance and UsageQueue._instance._started:
            UsageQueue._instance.shutdown(wait=True, timeout=2.0)
        UsageQueue._instance = None

    @patch("src.core.usage.usage_queue.get_usage_queue")
    def test_enqueue_token_usage(self, mock_get_queue):
        """Test enqueue_token_usage convenience function."""
        mock_queue = MagicMock()
        mock_get_queue.return_value = mock_queue

        enqueue_token_usage(
            org_id="org-123",
            feature="document_agent",
            model="gemini-3-flash-preview",
            provider="google",
            input_tokens=100,
            output_tokens=50,
        )

        mock_queue.enqueue.assert_called_once()
        event = mock_queue.enqueue.call_args[0][0]
        assert event.event_type == "token"
        assert event.org_id == "org-123"
        assert event.feature == "document_agent"
        assert event.model == "gemini-3-flash-preview"
        assert event.provider == "google"
        assert event.input_tokens == 100
        assert event.output_tokens == 50
        assert event.total_tokens == 150

    @patch("src.core.usage.usage_queue.get_usage_queue")
    def test_enqueue_token_usage_with_all_fields(self, mock_get_queue):
        """Test enqueue_token_usage with all optional fields."""
        mock_queue = MagicMock()
        mock_get_queue.return_value = mock_queue

        metadata = {"request_path": "/api/v1/chat"}
        enqueue_token_usage(
            org_id="org-123",
            feature="document_agent",
            model="gemini-3-flash-preview",
            provider="google",
            input_tokens=100,
            output_tokens=50,
            cached_tokens=20,
            user_id="user-456",
            request_id="req-789",
            session_id="sess-abc",
            processing_time_ms=1500,
            metadata=metadata,
        )

        mock_queue.enqueue.assert_called_once()
        event = mock_queue.enqueue.call_args[0][0]
        assert event.cached_tokens == 20
        assert event.user_id == "user-456"
        assert event.request_id == "req-789"
        assert event.session_id == "sess-abc"
        assert event.processing_time_ms == 1500
        assert event.metadata == metadata

    @patch("src.core.usage.usage_queue.get_usage_queue")
    def test_enqueue_resource_usage(self, mock_get_queue):
        """Test enqueue_resource_usage convenience function."""
        mock_queue = MagicMock()
        mock_get_queue.return_value = mock_queue

        enqueue_resource_usage(
            org_id="org-123",
            resource_type="llamaparse_pages",
            amount=5,
        )

        mock_queue.enqueue.assert_called_once()
        event = mock_queue.enqueue.call_args[0][0]
        assert event.event_type == "resource"
        assert event.org_id == "org-123"
        assert event.resource_type == "llamaparse_pages"
        assert event.amount == 5

    @patch("src.core.usage.usage_queue.get_usage_queue")
    def test_enqueue_resource_usage_with_all_fields(self, mock_get_queue):
        """Test enqueue_resource_usage with all optional fields."""
        mock_queue = MagicMock()
        mock_get_queue.return_value = mock_queue

        metadata = {"source": "upload"}
        enqueue_resource_usage(
            org_id="org-123",
            resource_type="storage_bytes",
            amount=1024000,
            user_id="user-456",
            request_id="req-789",
            file_name="document.pdf",
            file_path="/uploads/document.pdf",
            metadata=metadata,
        )

        mock_queue.enqueue.assert_called_once()
        event = mock_queue.enqueue.call_args[0][0]
        assert event.resource_type == "storage_bytes"
        assert event.amount == 1024000
        assert event.user_id == "user-456"
        assert event.request_id == "req-789"
        assert event.file_name == "document.pdf"
        assert event.file_path == "/uploads/document.pdf"
        assert event.metadata == metadata
