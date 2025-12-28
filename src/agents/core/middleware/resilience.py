"""
Resilience middleware for handling failures gracefully.

Provides:
- ModelRetry: Exponential backoff retry for model calls
- ModelFallback: Falls back to alternative model when primary fails
- ToolRetry: Retry logic for tool execution
"""

import logging
from typing import Any, Callable, Optional

from langchain.chat_models import init_chat_model
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

logger = logging.getLogger(__name__)


class ModelRetry:
    """
    Exponential backoff retry for model calls.

    Handles transient failures like network issues, rate limits, and timeouts.
    """

    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 10.0
    ):
        """
        Initialize model retry.

        Args:
            max_attempts: Maximum number of retry attempts
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
        """
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay

    def wrap(self, func: Callable) -> Callable:
        """
        Wrap a function with retry logic.

        Args:
            func: Function to wrap

        Returns:
            Wrapped function with retry logic
        """
        return retry(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential(
                multiplier=self.initial_delay,
                max=self.max_delay
            ),
            retry=retry_if_exception_type((
                TimeoutError,
                ConnectionError,
                OSError,
            )),
            reraise=True,
            before_sleep=before_sleep_log(logger, logging.WARNING)
        )(func)

    def create_retry_decorator(self):
        """Create a retry decorator for use with methods."""
        return retry(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential(
                multiplier=self.initial_delay,
                max=self.max_delay
            ),
            retry=retry_if_exception_type((
                TimeoutError,
                ConnectionError,
                OSError,
            )),
            reraise=True,
            before_sleep=before_sleep_log(logger, logging.WARNING)
        )


class ModelFallback:
    """
    Falls back to alternative model when primary fails.

    Provides resilience by using a secondary model (e.g., OpenAI)
    when the primary model (e.g., Gemini) is unavailable.
    """

    def __init__(
        self,
        primary_model: str,
        primary_provider: str,
        fallback_model: str = "gpt-4o-mini",
        fallback_provider: str = "openai",
        api_key: Optional[str] = None
    ):
        """
        Initialize model fallback.

        Args:
            primary_model: Primary model name
            primary_provider: Primary model provider
            fallback_model: Fallback model name
            fallback_provider: Fallback model provider
            api_key: Optional API key for fallback model
        """
        self.primary_model = primary_model
        self.primary_provider = primary_provider
        self.fallback_model = fallback_model
        self.fallback_provider = fallback_provider
        self.api_key = api_key
        self._fallback_llm = None
        self._fallback_agent = None

    @property
    def fallback_llm(self):
        """Lazy initialization of fallback LLM."""
        if self._fallback_llm is None:
            kwargs = {
                "model": self.fallback_model,
                "model_provider": self.fallback_provider,
            }
            if self.api_key:
                kwargs["api_key"] = self.api_key
            self._fallback_llm = init_chat_model(**kwargs)
            logger.info(f"Initialized fallback LLM: {self.fallback_model}")
        return self._fallback_llm

    async def execute_with_fallback(
        self,
        primary_func: Callable,
        fallback_func: Optional[Callable] = None,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute primary function, fall back on failure.

        Args:
            primary_func: Primary function to execute
            fallback_func: Optional fallback function (uses fallback_llm if not provided)
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Result from primary or fallback function
        """
        try:
            return await primary_func(*args, **kwargs)
        except Exception as e:
            logger.warning(
                f"Primary model ({self.primary_model}) failed: {e}. "
                f"Falling back to {self.fallback_model}"
            )

            if fallback_func:
                return await fallback_func(*args, **kwargs)
            else:
                # Use fallback LLM directly if no fallback function provided
                raise RuntimeError(
                    f"Fallback function not provided and primary failed: {e}"
                )

    def execute_with_fallback_sync(
        self,
        primary_func: Callable,
        fallback_func: Optional[Callable] = None,
        *args,
        **kwargs
    ) -> Any:
        """
        Synchronous version of execute_with_fallback.

        Args:
            primary_func: Primary function to execute
            fallback_func: Optional fallback function
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Result from primary or fallback function
        """
        try:
            return primary_func(*args, **kwargs)
        except Exception as e:
            logger.warning(
                f"Primary model ({self.primary_model}) failed: {e}. "
                f"Falling back to {self.fallback_model}"
            )

            if fallback_func:
                return fallback_func(*args, **kwargs)
            else:
                raise RuntimeError(
                    f"Fallback function not provided and primary failed: {e}"
                )


class ToolRetry:
    """
    Retry logic specifically for tool execution.

    Wraps tool's _run method with retry logic for transient failures.
    """

    def __init__(self, max_attempts: int = 2, delay: float = 1.0):
        """
        Initialize tool retry.

        Args:
            max_attempts: Maximum retry attempts
            delay: Base delay between retries in seconds
        """
        self.max_attempts = max_attempts
        self.delay = delay

    def wrap_tool(self, tool):
        """
        Wrap a tool's _run method with retry logic.

        Args:
            tool: LangChain tool to wrap

        Returns:
            Tool with wrapped _run method
        """
        original_run = tool._run

        @retry(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential(multiplier=self.delay, max=5),
            reraise=True,
            before_sleep=before_sleep_log(logger, logging.WARNING)
        )
        def wrapped_run(*args, **kwargs):
            return original_run(*args, **kwargs)

        tool._run = wrapped_run
        logger.debug(f"Wrapped tool '{tool.name}' with retry logic")
        return tool

    def wrap_tools(self, tools: list) -> list:
        """
        Wrap multiple tools with retry logic.

        Args:
            tools: List of tools to wrap

        Returns:
            List of wrapped tools
        """
        return [self.wrap_tool(tool) for tool in tools]
