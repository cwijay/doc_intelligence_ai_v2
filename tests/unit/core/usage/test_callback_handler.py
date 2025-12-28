"""Tests for TokenTrackingCallbackHandler."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from langchain_core.outputs import LLMResult, Generation

from src.core.usage.callback_handler import TokenTrackingCallbackHandler
from src.core.usage.schemas import TokenUsage
from src.core.usage.context import usage_context, clear_context


class TestTokenTrackingCallbackHandler:
    """Tests for TokenTrackingCallbackHandler."""

    def test_handler_initialization(self):
        """Test handler initializes with correct parameters."""
        handler = TokenTrackingCallbackHandler(
            org_id="org-123",
            feature="document_agent",
            user_id="user-456",
            session_id="session-789",
        )

        assert handler.org_id == "org-123"
        assert handler.feature == "document_agent"
        assert handler.user_id == "user-456"
        assert handler.session_id == "session-789"
        assert handler.total_usage.total_tokens == 0

    def test_handler_defaults(self):
        """Test handler with minimal parameters."""
        handler = TokenTrackingCallbackHandler(
            org_id="org-123",
            feature="sheets_agent",
        )

        assert handler.org_id == "org-123"
        assert handler.feature == "sheets_agent"
        assert handler.user_id is None
        assert handler.session_id is None

    def test_on_llm_end_extracts_tokens(self):
        """Test on_llm_end extracts tokens from response."""
        handler = TokenTrackingCallbackHandler(
            org_id="org-123",
            feature="document_agent",
        )

        mock_result = LLMResult(
            generations=[[Generation(text="Hello world")]],
            llm_output={
                "token_usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
                "model_name": "gpt-4o",
            },
        )

        # Patch _enqueue_usage to avoid queue logging
        with patch.object(handler, "_enqueue_usage"):
            handler.on_llm_end(mock_result)

        assert handler.total_usage.input_tokens == 100
        assert handler.total_usage.output_tokens == 50
        assert handler.total_usage.total_tokens == 150

    def test_on_llm_end_accumulates_across_calls(self):
        """Test that multiple on_llm_end calls accumulate tokens."""
        handler = TokenTrackingCallbackHandler(
            org_id="org-123",
            feature="document_agent",
        )

        result1 = LLMResult(
            generations=[[Generation(text="Hello")]],
            llm_output={
                "token_usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
            },
        )

        result2 = LLMResult(
            generations=[[Generation(text="World")]],
            llm_output={
                "token_usage": {
                    "prompt_tokens": 200,
                    "completion_tokens": 100,
                    "total_tokens": 300,
                },
            },
        )

        with patch.object(handler, "_enqueue_usage"):
            handler.on_llm_end(result1)
            handler.on_llm_end(result2)

        assert handler.total_usage.input_tokens == 300  # 100 + 200
        assert handler.total_usage.output_tokens == 150  # 50 + 100
        assert handler.total_usage.total_tokens == 450  # 150 + 300

    def test_on_llm_end_handles_missing_usage(self):
        """Test on_llm_end handles missing usage gracefully."""
        handler = TokenTrackingCallbackHandler(
            org_id="org-123",
            feature="document_agent",
        )

        mock_result = LLMResult(
            generations=[[Generation(text="Hello world")]],
            llm_output=None,
        )

        with patch.object(handler, "_enqueue_usage"):
            # Should not raise
            handler.on_llm_end(mock_result)

        # Returns empty TokenUsage when llm_output is None
        assert handler.total_usage.total_tokens >= 0

    def test_get_total_usage(self):
        """Test getting total accumulated usage."""
        handler = TokenTrackingCallbackHandler(
            org_id="org-123",
            feature="document_agent",
        )

        result = LLMResult(
            generations=[[Generation(text="Hello")]],
            llm_output={
                "token_usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
            },
        )

        with patch.object(handler, "_enqueue_usage"):
            handler.on_llm_end(result)

        usage = handler.total_usage

        assert isinstance(usage, TokenUsage)
        assert usage.total_tokens == 150

    def test_reset(self):
        """Test resetting accumulated usage."""
        handler = TokenTrackingCallbackHandler(
            org_id="org-123",
            feature="document_agent",
        )

        result = LLMResult(
            generations=[[Generation(text="Hello")]],
            llm_output={
                "token_usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
            },
        )

        with patch.object(handler, "_enqueue_usage"):
            handler.on_llm_end(result)

        assert handler.total_usage.total_tokens == 150

        handler.reset()

        assert handler.total_usage.total_tokens == 0
        assert handler.total_usage.input_tokens == 0
        assert handler.total_usage.output_tokens == 0

    def test_call_count_tracking(self):
        """Test LLM call counting."""
        handler = TokenTrackingCallbackHandler(
            org_id="org-123",
            feature="document_agent",
        )

        assert handler.call_count == 0

        # Simulate LLM start
        handler.on_llm_start({}, ["prompt 1"])
        assert handler.call_count == 1

        handler.on_llm_start({}, ["prompt 2"])
        assert handler.call_count == 2

        handler.reset()
        assert handler.call_count == 0


class TestTokenTrackingCallbackHandlerContextMode:
    """Tests for TokenTrackingCallbackHandler with use_context=True."""

    def setup_method(self):
        """Clean up context before each test."""
        clear_context()

    def teardown_method(self):
        """Clean up context after each test."""
        clear_context()

    def test_handler_context_mode_initialization(self):
        """Test handler initializes with use_context=True."""
        handler = TokenTrackingCallbackHandler(
            org_id="",  # Empty - will use context
            feature="document_agent",
            use_context=True,
        )

        assert handler.use_context is True
        assert handler.org_id == ""

    def test_get_effective_context_with_context(self):
        """Test _get_effective_context returns context values when set."""
        handler = TokenTrackingCallbackHandler(
            org_id="",
            feature="default_feature",
            use_context=True,
        )

        with usage_context(
            org_id="context-org",
            feature="context_feature",
            user_id="context-user",
            session_id="context-session",
        ):
            org_id, feature, user_id, session_id, _, _ = handler._get_effective_context()

            assert org_id == "context-org"
            assert feature == "context_feature"
            assert user_id == "context-user"
            assert session_id == "context-session"

    def test_get_effective_context_without_context(self):
        """Test _get_effective_context falls back to instance values."""
        handler = TokenTrackingCallbackHandler(
            org_id="instance-org",
            feature="instance_feature",
            user_id="instance-user",
            use_context=True,
        )

        # No context set - should use instance values
        org_id, feature, user_id, session_id, _, _ = handler._get_effective_context()

        assert org_id == "instance-org"
        assert feature == "instance_feature"
        assert user_id == "instance-user"

    def test_get_effective_context_disabled(self):
        """Test _get_effective_context ignores context when use_context=False."""
        handler = TokenTrackingCallbackHandler(
            org_id="instance-org",
            feature="instance_feature",
            use_context=False,  # Disabled
        )

        with usage_context(
            org_id="context-org",
            feature="context_feature",
        ):
            org_id, feature, _, _, _, _ = handler._get_effective_context()

            # Should use instance values, not context
            assert org_id == "instance-org"
            assert feature == "instance_feature"

    def test_enqueue_usage_with_context(self):
        """Test _enqueue_usage uses context for org_id."""
        handler = TokenTrackingCallbackHandler(
            org_id="",
            feature="document_agent",
            use_context=True,
        )

        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            model="gpt-4o",
            provider="openai",
        )

        with usage_context(
            org_id="queue-test-org",
            feature="queue_feature",
        ):
            with patch("src.core.usage.usage_queue.enqueue_token_usage") as mock_enqueue:
                handler._enqueue_usage(usage)

                mock_enqueue.assert_called_once()
                call_kwargs = mock_enqueue.call_args.kwargs
                assert call_kwargs["org_id"] == "queue-test-org"
                assert call_kwargs["feature"] == "queue_feature"

    def test_enqueue_usage_skipped_without_org_id(self):
        """Test _enqueue_usage skips when no org_id in context."""
        handler = TokenTrackingCallbackHandler(
            org_id="",  # No org_id
            feature="document_agent",
            use_context=True,
        )

        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        )

        # No context set, no org_id
        with patch("src.core.usage.usage_queue.enqueue_token_usage") as mock_enqueue:
            handler._enqueue_usage(usage)

            # Should not be called because org_id is empty
            mock_enqueue.assert_not_called()

    def test_on_llm_end_with_context_mode(self):
        """Test on_llm_end uses context in context mode."""
        handler = TokenTrackingCallbackHandler(
            org_id="",
            feature="document_agent",
            use_context=True,
        )

        mock_result = LLMResult(
            generations=[[Generation(text="Hello world")]],
            llm_output={
                "token_usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
                "model_name": "gpt-4o",
            },
        )

        with usage_context(
            org_id="llm-end-org",
            feature="llm_end_feature",
        ):
            with patch("src.core.usage.usage_queue.enqueue_token_usage") as mock_enqueue:
                handler.on_llm_end(mock_result)

                mock_enqueue.assert_called_once()
                call_kwargs = mock_enqueue.call_args.kwargs
                assert call_kwargs["org_id"] == "llm-end-org"
                assert call_kwargs["input_tokens"] == 100
                assert call_kwargs["output_tokens"] == 50
