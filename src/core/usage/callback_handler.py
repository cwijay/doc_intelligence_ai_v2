"""
LangChain callback handler for automatic token tracking.

Intercepts on_llm_end() to capture actual token usage from LLM responses.

Supports two modes:
1. Explicit org_id: Passed at construction (per-request handler)
2. Context-based: Uses thread-local UsageContext (singleton handler with use_context=True)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from .schemas import TokenUsage
from .token_extractors import extract_from_langchain_response

logger = logging.getLogger(__name__)


class TokenTrackingCallbackHandler(BaseCallbackHandler):
    """
    LangChain callback handler for automatic token tracking.

    Intercepts on_llm_end() to extract actual token counts from
    LLM responses and log them via the usage queue.

    Supports two modes:
    1. Explicit org_id: Passed at construction (per-request handler)
    2. Context-based: Uses thread-local UsageContext (singleton handler)

    Usage (explicit mode):
        handler = TokenTrackingCallbackHandler(
            org_id="org_123",
            feature="document_agent",
            user_id="user_456",
        )
        llm = init_chat_model(..., callbacks=[handler])

    Usage (context mode):
        handler = TokenTrackingCallbackHandler(
            org_id="",  # Will use context
            feature="document_agent",
            use_context=True,
        )
        llm = init_chat_model(..., callbacks=[handler])

        # Later, at request time:
        with usage_context(org_id="org_123", feature="document_agent"):
            result = agent.invoke(...)
    """

    def __init__(
        self,
        org_id: str = "",
        feature: str = "unknown",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        use_context: bool = False,
    ):
        """
        Initialize callback handler.

        Args:
            org_id: Organization ID for usage tracking (can be empty if use_context=True)
            feature: Feature name (document_agent, sheets_agent, rag_search)
            user_id: Optional user ID
            session_id: Optional session ID
            request_id: Optional request ID for deduplication
            metadata: Optional additional metadata
            use_context: If True, get org_id and metadata from thread-local context
        """
        super().__init__()
        self.org_id = org_id
        self.feature = feature
        self.user_id = user_id
        self.session_id = session_id
        self.request_id = request_id
        self.metadata = metadata or {}
        self.use_context = use_context

        # Accumulate usage across multiple LLM calls
        self._accumulated_usage = TokenUsage()
        self._call_count = 0

    def _get_effective_context(self) -> Tuple[str, str, Optional[str], Optional[str], Optional[str], Dict]:
        """
        Get org_id and metadata from context or instance.

        Returns:
            Tuple of (org_id, feature, user_id, session_id, request_id, metadata)
        """
        if self.use_context:
            from .context import get_current_context

            ctx = get_current_context()
            if ctx:
                return (
                    ctx.org_id,
                    ctx.feature or self.feature,
                    ctx.user_id or self.user_id,
                    ctx.session_id or self.session_id,
                    ctx.request_id or self.request_id,
                    {**self.metadata, **(ctx.metadata or {})},
                )

        # Fall back to instance values
        return (
            self.org_id,
            self.feature,
            self.user_id,
            self.session_id,
            self.request_id,
            self.metadata,
        )

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any,
    ) -> None:
        """Called when LLM starts generating."""
        self._call_count += 1
        org_id, feature, *_ = self._get_effective_context()
        logger.debug(f"LLM call {self._call_count} started for {feature} (org={org_id or 'unknown'})")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """
        Called when LLM completes generation.

        Extracts actual token usage and enqueues for logging.
        """
        try:
            # Extract token usage from LangChain response
            usage = extract_from_langchain_response(response)

            # Try to get model info from response
            llm_output = response.llm_output or {}
            model_name = llm_output.get("model_name", "")

            if not usage.model:
                usage.model = model_name

            # Determine provider from model name or llm_output
            if not usage.provider:
                if "gpt" in model_name.lower() or "openai" in str(llm_output).lower():
                    usage.provider = "openai"
                elif "gemini" in model_name.lower() or "google" in str(llm_output).lower():
                    usage.provider = "google"

            # Accumulate usage
            self._accumulated_usage = self._accumulated_usage + usage

            # Enqueue for logging if we have actual tokens
            if usage.total_tokens > 0:
                self._enqueue_usage(usage)

                org_id, feature, *_ = self._get_effective_context()
                logger.debug(
                    f"Tracked token usage: org={org_id or 'unknown'}, feature={feature}, "
                    f"tokens={usage.total_tokens} (in={usage.input_tokens}, out={usage.output_tokens})"
                )

        except Exception as e:
            # Never fail the LLM call due to tracking errors
            logger.warning(f"Failed to track token usage: {e}")

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """Called on LLM error."""
        _, feature, *_ = self._get_effective_context()
        logger.debug(f"LLM error for {feature}: {error}")

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Called when chain starts."""
        pass

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Called when chain ends."""
        pass

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Called when tool starts."""
        pass

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Called when tool ends."""
        pass

    def _enqueue_usage(self, usage: TokenUsage) -> None:
        """
        Enqueue usage for background processing (sync-safe).

        Uses the usage queue which is thread-safe and doesn't require
        an event loop, making it safe to call from sync contexts.
        """
        org_id, feature, user_id, session_id, request_id, metadata = self._get_effective_context()

        if not org_id:
            logger.debug("No org_id in context, skipping token logging")
            return

        try:
            from .usage_queue import enqueue_token_usage

            enqueue_token_usage(
                org_id=org_id,
                feature=feature,
                model=usage.model or "unknown",
                provider=usage.provider or "unknown",
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cached_tokens=usage.cached_tokens,
                user_id=user_id,
                request_id=request_id,
                session_id=session_id,
                metadata=metadata,
            )
        except Exception as e:
            logger.warning(f"Failed to enqueue token usage: {e}")

    @property
    def total_usage(self) -> TokenUsage:
        """Get total accumulated usage across all LLM calls."""
        return self._accumulated_usage

    @property
    def call_count(self) -> int:
        """Get number of LLM calls tracked."""
        return self._call_count

    def reset(self) -> None:
        """Reset accumulated usage counter."""
        self._accumulated_usage = TokenUsage()
        self._call_count = 0


__all__ = [
    "TokenTrackingCallbackHandler",
]
