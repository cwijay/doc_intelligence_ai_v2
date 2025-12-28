"""
Centralized audit queue with dedicated event loop.

Prevents connection pool proliferation by using a single persistent
event loop for all background audit operations.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.core.queues import BackgroundQueue

logger = logging.getLogger(__name__)


@dataclass
class AuditEvent:
    """Audit event to be logged."""
    event_type: str
    file_name: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    organization_id: Optional[str] = None
    document_hash: Optional[str] = None
    job_id: Optional[str] = None


class AuditQueue(BackgroundQueue[AuditEvent]):
    """
    Thread-safe audit queue with dedicated event loop.

    Uses a single background thread with a persistent event loop,
    limiting database connection pools to exactly 2 (main + audit).
    """

    def _get_queue_name(self) -> str:
        """Return queue name for logging."""
        return "audit-queue"

    async def _process_event(self, event: AuditEvent) -> None:
        """Process a single audit event."""
        from src.db.repositories.audit_repository import log_event

        await log_event(
            event_type=event.event_type,
            file_name=event.file_name,
            details=event.details,
            organization_id=event.organization_id,
            document_hash=event.document_hash,
            job_id=event.job_id,
        )


# Singleton accessor
def get_audit_queue() -> AuditQueue:
    """Get singleton audit queue instance."""
    return AuditQueue.get_instance()


def enqueue_audit_event(
    event_type: str,
    file_name: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    organization_id: Optional[str] = None,
    document_hash: Optional[str] = None,
    job_id: Optional[str] = None,
):
    """Convenience function to enqueue an audit event."""
    event = AuditEvent(
        event_type=event_type,
        file_name=file_name,
        details=details,
        organization_id=organization_id,
        document_hash=document_hash,
        job_id=job_id,
    )
    get_audit_queue().enqueue(event)
