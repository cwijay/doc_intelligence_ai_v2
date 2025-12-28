"""
Centralized usage queue with dedicated event loop.

Prevents connection pool proliferation by using a single persistent
event loop for all background usage tracking operations.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Literal, Optional

from src.core.queues import BackgroundQueue

logger = logging.getLogger(__name__)


@dataclass
class UsageEvent:
    """Usage event to be logged."""

    event_type: Literal["token", "resource"]
    org_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Token usage fields
    feature: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0

    # Resource usage fields
    resource_type: Optional[str] = None  # llamaparse_pages, file_search_queries, storage_bytes
    amount: int = 0
    file_name: Optional[str] = None
    file_path: Optional[str] = None

    # Common fields
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    processing_time_ms: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class UsageQueue(BackgroundQueue[UsageEvent]):
    """
    Thread-safe usage queue with dedicated event loop.

    Uses a single background thread with a persistent event loop,
    limiting database connection pools to exactly 3 (main + audit + usage).
    """

    def _get_queue_name(self) -> str:
        """Return queue name for logging."""
        return "usage-queue"

    async def _process_event(self, event: UsageEvent) -> None:
        """Process a single usage event."""
        from .schemas import TokenUsage
        from .service import get_usage_service

        service = get_usage_service()

        if event.event_type == "token":
            usage = TokenUsage(
                input_tokens=event.input_tokens,
                output_tokens=event.output_tokens,
                total_tokens=event.total_tokens,
                cached_tokens=event.cached_tokens,
                provider=event.provider,
                model=event.model,
            )
            await service.log_token_usage(
                org_id=event.org_id,
                feature=event.feature or "unknown",
                usage=usage,
                user_id=event.user_id,
                request_id=event.request_id,
                session_id=event.session_id,
                metadata=event.metadata,
                processing_time_ms=event.processing_time_ms,
            )
        elif event.event_type == "resource":
            await service.log_resource_usage(
                org_id=event.org_id,
                resource_type=event.resource_type or "unknown",
                amount=event.amount,
                user_id=event.user_id,
                request_id=event.request_id,
                file_name=event.file_name,
                file_path=event.file_path,
                metadata=event.metadata,
            )


# Singleton accessor
def get_usage_queue() -> UsageQueue:
    """Get singleton usage queue instance."""
    return UsageQueue.get_instance()


def enqueue_token_usage(
    org_id: str,
    feature: str,
    model: str,
    provider: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
    session_id: Optional[str] = None,
    processing_time_ms: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Convenience function to enqueue a token usage event."""
    event = UsageEvent(
        event_type="token",
        org_id=org_id,
        feature=feature,
        model=model,
        provider=provider,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cached_tokens=cached_tokens,
        user_id=user_id,
        request_id=request_id,
        session_id=session_id,
        processing_time_ms=processing_time_ms,
        metadata=metadata,
    )
    get_usage_queue().enqueue(event)


def enqueue_resource_usage(
    org_id: str,
    resource_type: str,
    amount: int,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
    file_name: Optional[str] = None,
    file_path: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Convenience function to enqueue a resource usage event."""
    event = UsageEvent(
        event_type="resource",
        org_id=org_id,
        resource_type=resource_type,
        amount=amount,
        user_id=user_id,
        request_id=request_id,
        file_name=file_name,
        file_path=file_path,
        metadata=metadata,
    )
    get_usage_queue().enqueue(event)
