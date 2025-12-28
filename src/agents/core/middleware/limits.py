"""
Call limits middleware for DocumentAgent.

Provides:
- CallLimitTracker: Tracks and enforces limits on model and tool calls
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class CallLimitExceeded(Exception):
    """Raised when call limits are exceeded."""

    def __init__(self, call_type: str, limit: int, current: int):
        self.call_type = call_type
        self.limit = limit
        self.current = current
        super().__init__(
            f"{call_type} call limit exceeded: {current}/{limit}"
        )


class CallLimitTracker:
    """
    Tracks and enforces limits on model and tool calls.

    Prevents runaway agent loops by limiting the number of calls
    that can be made in a single run.
    """

    def __init__(
        self,
        model_call_limit: int = 15,
        tool_call_limit: int = 10
    ):
        """
        Initialize call limit tracker.

        Args:
            model_call_limit: Maximum model calls per run
            tool_call_limit: Maximum tool calls per run
        """
        self.model_call_limit = model_call_limit
        self.tool_call_limit = tool_call_limit
        self._model_calls = 0
        self._tool_calls = 0

    def reset(self):
        """Reset all counters for a new run."""
        self._model_calls = 0
        self._tool_calls = 0
        logger.debug("Call limit counters reset")

    @property
    def model_calls(self) -> int:
        """Get current model call count."""
        return self._model_calls

    @property
    def tool_calls(self) -> int:
        """Get current tool call count."""
        return self._tool_calls

    def check_model_limit(self) -> bool:
        """
        Check if model call limit would be exceeded.

        Returns:
            True if within limits, False otherwise
        """
        return self._model_calls < self.model_call_limit

    def check_tool_limit(self) -> bool:
        """
        Check if tool call limit would be exceeded.

        Returns:
            True if within limits, False otherwise
        """
        return self._tool_calls < self.tool_call_limit

    def increment_model_calls(self, raise_on_limit: bool = True) -> int:
        """
        Increment model call counter.

        Args:
            raise_on_limit: If True, raise exception when limit exceeded

        Returns:
            Current model call count after increment

        Raises:
            CallLimitExceeded: If limit exceeded and raise_on_limit is True
        """
        self._model_calls += 1
        logger.debug(
            f"Model calls: {self._model_calls}/{self.model_call_limit}"
        )

        if self._model_calls > self.model_call_limit:
            if raise_on_limit:
                raise CallLimitExceeded(
                    "Model", self.model_call_limit, self._model_calls
                )
            logger.warning(
                f"Model call limit exceeded: {self._model_calls}/{self.model_call_limit}"
            )

        return self._model_calls

    def increment_tool_calls(self, raise_on_limit: bool = True) -> int:
        """
        Increment tool call counter.

        Args:
            raise_on_limit: If True, raise exception when limit exceeded

        Returns:
            Current tool call count after increment

        Raises:
            CallLimitExceeded: If limit exceeded and raise_on_limit is True
        """
        self._tool_calls += 1
        logger.debug(
            f"Tool calls: {self._tool_calls}/{self.tool_call_limit}"
        )

        if self._tool_calls > self.tool_call_limit:
            if raise_on_limit:
                raise CallLimitExceeded(
                    "Tool", self.tool_call_limit, self._tool_calls
                )
            logger.warning(
                f"Tool call limit exceeded: {self._tool_calls}/{self.tool_call_limit}"
            )

        return self._tool_calls

    def get_remaining(self) -> dict:
        """
        Get remaining calls for each type.

        Returns:
            Dictionary with remaining model and tool calls
        """
        return {
            "model_calls_remaining": max(
                0, self.model_call_limit - self._model_calls
            ),
            "tool_calls_remaining": max(
                0, self.tool_call_limit - self._tool_calls
            )
        }

    def get_stats(self) -> dict:
        """
        Get current call statistics.

        Returns:
            Dictionary with current counts and limits
        """
        return {
            "model_calls": self._model_calls,
            "model_call_limit": self.model_call_limit,
            "tool_calls": self._tool_calls,
            "tool_call_limit": self.tool_call_limit,
            **self.get_remaining()
        }

    def create_callback_handler(self):
        """
        Create a LangChain callback handler for automatic tracking.

        Returns:
            CallLimitCallbackHandler instance
        """
        return CallLimitCallbackHandler(self)


class CallLimitCallbackHandler:
    """
    LangChain callback handler for automatic call tracking.

    Integrates with LangChain's callback system to automatically
    track model and tool calls.
    """

    def __init__(self, tracker: CallLimitTracker):
        """
        Initialize callback handler.

        Args:
            tracker: CallLimitTracker instance to use
        """
        self.tracker = tracker

    def on_llm_start(self, *args, **kwargs):
        """Called when LLM starts - increment model call counter."""
        self.tracker.increment_model_calls()

    def on_tool_start(self, *args, **kwargs):
        """Called when tool starts - increment tool call counter."""
        self.tracker.increment_tool_calls()

    def on_chain_start(self, *args, **kwargs):
        """Called when chain starts - no action needed."""
        pass

    def on_chain_end(self, *args, **kwargs):
        """Called when chain ends - no action needed."""
        pass

    def on_llm_end(self, *args, **kwargs):
        """Called when LLM ends - no action needed."""
        pass

    def on_tool_end(self, *args, **kwargs):
        """Called when tool ends - no action needed."""
        pass

    def on_chain_error(self, *args, **kwargs):
        """Called on chain error - no action needed."""
        pass

    def on_llm_error(self, *args, **kwargs):
        """Called on LLM error - no action needed."""
        pass

    def on_tool_error(self, *args, **kwargs):
        """Called on tool error - no action needed."""
        pass
