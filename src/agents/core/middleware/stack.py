"""
Middleware stack builder for DocumentAgent.

Composes multiple middleware components into a unified stack
that can be easily integrated with the agent.
"""

import logging
import os
from typing import List, Optional

from langchain_core.tools import BaseTool

from .config import MiddlewareConfig

# Default model from environment (consistent with DocumentAgentConfig)
DEFAULT_DOCUMENT_AGENT_MODEL = os.getenv("DOCUMENT_AGENT_MODEL", "gemini-3-flash-preview")
from .limits import CallLimitTracker
from .resilience import ModelFallback, ModelRetry, ToolRetry
from .safety import PIIDetector, PIIStrategy
from .tool_selector import LLMToolSelector

logger = logging.getLogger(__name__)


class MiddlewareStack:
    """
    Composes and manages all middleware components.

    Provides a unified interface for applying middleware to the agent.
    """

    def __init__(self, config: Optional[MiddlewareConfig] = None):
        """
        Initialize middleware stack.

        Args:
            config: Middleware configuration (uses defaults if not provided)
        """
        self.config = config or MiddlewareConfig()
        self._initialized = False

        # Components (lazily initialized)
        self._tool_selector: Optional[LLMToolSelector] = None
        self._model_retry: Optional[ModelRetry] = None
        self._tool_retry: Optional[ToolRetry] = None
        self._model_fallback: Optional[ModelFallback] = None
        self._call_limiter: Optional[CallLimitTracker] = None
        self._pii_detector: Optional[PIIDetector] = None

    def initialize(
        self,
        primary_model: Optional[str] = None,
        primary_provider: str = "google_genai"
    ):
        """
        Initialize all middleware components based on config.

        Args:
            primary_model: Primary model name for fallback config.
                           Defaults to DOCUMENT_AGENT_MODEL env var.
            primary_provider: Primary model provider for fallback config
        """
        primary_model = primary_model or DEFAULT_DOCUMENT_AGENT_MODEL
        if not self.config.enabled:
            logger.info("Middleware disabled via configuration")
            self._initialized = True
            return

        logger.info("Initializing middleware stack...")

        # Tool Selector
        if self.config.tool_selector_enabled:
            self._tool_selector = LLMToolSelector(
                model=self.config.tool_selector_model,
                provider="google_genai",
                max_tools=self.config.tool_selector_max_tools
            )
            logger.debug("Tool selector initialized")

        # Model Retry
        if self.config.model_retry_enabled:
            self._model_retry = ModelRetry(
                max_attempts=self.config.model_retry_max_attempts,
                initial_delay=self.config.model_retry_initial_delay,
                max_delay=self.config.model_retry_max_delay
            )
            logger.debug("Model retry initialized")

        # Tool Retry
        if self.config.tool_retry_enabled:
            self._tool_retry = ToolRetry(
                max_attempts=self.config.tool_retry_max_attempts,
                delay=self.config.tool_retry_delay
            )
            logger.debug("Tool retry initialized")

        # Model Fallback
        if self.config.fallback_enabled:
            self._model_fallback = ModelFallback(
                primary_model=primary_model,
                primary_provider=primary_provider,
                fallback_model=self.config.fallback_model,
                fallback_provider=self.config.fallback_provider
            )
            logger.debug("Model fallback initialized")

        # Call Limits
        self._call_limiter = CallLimitTracker(
            model_call_limit=self.config.model_call_limit,
            tool_call_limit=self.config.tool_call_limit
        )
        logger.debug("Call limiter initialized")

        # PII Detection
        if self.config.pii_enabled:
            try:
                strategy = PIIStrategy(self.config.pii_strategy)
            except ValueError:
                logger.warning(
                    f"Invalid PII strategy '{self.config.pii_strategy}', "
                    "defaulting to 'redact'"
                )
                strategy = PIIStrategy.REDACT

            self._pii_detector = PIIDetector(strategy=strategy)
            logger.debug("PII detector initialized")

        self._initialized = True
        logger.info("Middleware stack initialized successfully")

    @property
    def is_enabled(self) -> bool:
        """Check if middleware is enabled."""
        return self.config.enabled

    @property
    def tool_selector(self) -> Optional[LLMToolSelector]:
        """Get tool selector component."""
        return self._tool_selector

    @property
    def model_retry(self) -> Optional[ModelRetry]:
        """Get model retry component."""
        return self._model_retry

    @property
    def tool_retry(self) -> Optional[ToolRetry]:
        """Get tool retry component."""
        return self._tool_retry

    @property
    def model_fallback(self) -> Optional[ModelFallback]:
        """Get model fallback component."""
        return self._model_fallback

    @property
    def call_limiter(self) -> Optional[CallLimitTracker]:
        """Get call limiter component."""
        return self._call_limiter

    @property
    def pii_detector(self) -> Optional[PIIDetector]:
        """Get PII detector component."""
        return self._pii_detector

    def select_tools(
        self,
        query: str,
        available_tools: List[BaseTool],
        context: Optional[str] = None
    ) -> List[BaseTool]:
        """
        Select relevant tools for query.

        Args:
            query: User query
            available_tools: All available tools
            context: Optional context

        Returns:
            Selected subset of tools (or all if selector disabled)
        """
        if not self._tool_selector:
            return available_tools

        return self._tool_selector.select_tools(query, available_tools, context)

    def wrap_tools(self, tools: List[BaseTool]) -> List[BaseTool]:
        """
        Wrap tools with retry logic.

        Args:
            tools: Tools to wrap

        Returns:
            Wrapped tools (or original if retry disabled)
        """
        if not self._tool_retry:
            return tools

        return self._tool_retry.wrap_tools(tools)

    def process_input(self, text: str) -> str:
        """
        Process input text through safety middleware.

        Args:
            text: Input text

        Returns:
            Processed text (PII handled according to strategy)
        """
        if not self._pii_detector:
            return text

        processed, matches = self._pii_detector.process(text)
        if matches:
            logger.info(f"Processed {len(matches)} PII instances in input")
        return processed

    def process_output(self, text: str) -> str:
        """
        Process output text through safety middleware.

        Args:
            text: Output text

        Returns:
            Processed text (PII handled according to strategy)
        """
        if not self._pii_detector:
            return text

        processed, matches = self._pii_detector.process(text)
        if matches:
            logger.info(f"Processed {len(matches)} PII instances in output")
        return processed

    def reset_limits(self):
        """Reset call limit counters for a new run."""
        if self._call_limiter:
            self._call_limiter.reset()

    def get_callback_handler(self):
        """
        Get LangChain callback handler for automatic tracking.

        Returns:
            Callback handler or None if call limiting disabled
        """
        if self._call_limiter:
            return self._call_limiter.create_callback_handler()
        return None

    def get_stats(self) -> dict:
        """
        Get middleware statistics.

        Returns:
            Dictionary with component statuses and stats
        """
        stats = {
            "enabled": self.config.enabled,
            "initialized": self._initialized,
            "components": {
                "tool_selector": self._tool_selector is not None,
                "model_retry": self._model_retry is not None,
                "tool_retry": self._tool_retry is not None,
                "model_fallback": self._model_fallback is not None,
                "call_limiter": self._call_limiter is not None,
                "pii_detector": self._pii_detector is not None,
            }
        }

        if self._call_limiter:
            stats["call_limits"] = self._call_limiter.get_stats()

        return stats


def create_middleware_stack(
    config: Optional[MiddlewareConfig] = None,
    primary_model: Optional[str] = None,
    primary_provider: str = "google_genai"
) -> MiddlewareStack:
    """
    Factory function to create and initialize a middleware stack.

    Args:
        config: Optional middleware configuration
        primary_model: Primary model name (defaults to DOCUMENT_AGENT_MODEL env var)
        primary_provider: Primary model provider

    Returns:
        Initialized MiddlewareStack
    """
    stack = MiddlewareStack(config)
    stack.initialize(primary_model or DEFAULT_DOCUMENT_AGENT_MODEL, primary_provider)
    return stack
