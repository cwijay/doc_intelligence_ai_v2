"""
Request context for token tracking using contextvars.

Allows the callback handler to access org_id and other request metadata
even when the LLM was initialized as a singleton without the org_id.

Uses Python's contextvars module which properly propagates context across
asyncio.loop.run_in_executor() calls, unlike threading.local().

Usage:
    from src.core.usage.context import usage_context

    with usage_context(
        org_id="org_123",
        feature="document_agent",
        user_id="user_456",
    ):
        # Any LLM calls here will have access to the context
        # Even when run in executor threads via asyncio
        result = agent.invoke(...)
"""

import logging
from contextvars import ContextVar, Token
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ContextVar for request context - propagates across asyncio executor calls
_context: ContextVar[Optional["UsageContext"]] = ContextVar("usage_context", default=None)


@dataclass
class UsageContext:
    """Usage tracking context for a request."""

    org_id: str
    feature: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)


def get_current_context() -> Optional[UsageContext]:
    """
    Get the current request context.

    Works across asyncio executor threads when set via usage_context().

    Returns:
        UsageContext if set, None otherwise
    """
    return _context.get()


def set_context(context: UsageContext) -> Token:
    """
    Set the current request context.

    Args:
        context: UsageContext to set

    Returns:
        Token that can be used to restore previous value
    """
    return _context.set(context)


def clear_context() -> None:
    """Clear the current request context."""
    _context.set(None)


@contextmanager
def usage_context(
    org_id: str,
    feature: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    request_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Context manager for setting usage context during a request.

    This sets up a context that the TokenTrackingCallbackHandler can access
    to get the org_id and other metadata for logging.

    Uses contextvars which properly propagates context across
    asyncio.loop.run_in_executor() calls, enabling token tracking in
    executor threads.

    Args:
        org_id: Organization ID for usage tracking
        feature: Feature name (document_agent, sheets_agent, rag_search)
        user_id: Optional user ID
        session_id: Optional session ID
        request_id: Optional request ID for deduplication
        metadata: Optional additional metadata

    Yields:
        UsageContext: The created context

    Example:
        with usage_context(org_id="org_123", feature="document_agent"):
            result = await agent.process_request(request)
    """
    ctx = UsageContext(
        org_id=org_id,
        feature=feature,
        user_id=user_id,
        session_id=session_id,
        request_id=request_id,
        metadata=metadata or {},
    )
    token = _context.set(ctx)  # Returns token for restoration
    try:
        yield ctx
    finally:
        _context.reset(token)  # Restores previous value (handles nested contexts)


__all__ = [
    "UsageContext",
    "usage_context",
    "get_current_context",
    "set_context",
    "clear_context",
]
