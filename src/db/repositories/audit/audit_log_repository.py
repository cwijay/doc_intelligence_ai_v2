"""Audit log repository - Event logging and audit trail queries.

Multi-tenancy: All operations are scoped by organization_id for tenant isolation.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy import select, and_, desc
from sqlalchemy.exc import IntegrityError

from ...models import AuditLog
from ...connection import db
from ...utils import with_db_retry

from .document_repository import get_document_by_name
from .job_repository import get_jobs_by_document

logger = logging.getLogger(__name__)


# =============================================================================
# AUDIT LOGGING
# =============================================================================


@with_db_retry
async def log_event(
    event_type: str,
    document_hash: Optional[str] = None,
    file_name: Optional[str] = None,
    job_id: Optional[str] = None,
    details: Optional[dict] = None,
    organization_id: Optional[str] = None,
    action: Optional[str] = None,
    entity_type: Optional[str] = None,
    entity_id: Optional[str] = None,
):
    """
    Add audit log entry.

    Multi-tenancy: Associates event with organization_id for tenant isolation.
    If organization_id is not provided, the event is logged at DEBUG level and skipped
    to avoid database constraint errors during development/testing.

    Args:
        event_type: Type of event (parse_started, parse_completed, cache_hit, error)
        document_hash: SHA-256 hash of the document
        file_name: Name of the document file
        job_id: Associated job ID
        details: Additional event details
        organization_id: Organization ID for tenant isolation (required for DB insert)
        action: Audit action (CREATE, UPDATE, DELETE, etc.) - auto-derived if not provided
        entity_type: Entity type (ORGANIZATION, USER, FOLDER, DOCUMENT) - defaults to DOCUMENT
        entity_id: Entity identifier - defaults to document_hash or "unknown"
    """
    async with db.session() as session:
        if session is None:
            # Database disabled - skip logging
            return

        # Skip audit logging if organization_id is not provided
        # This prevents NOT NULL constraint violations during development/testing
        if organization_id is None:
            logger.debug(
                f"Skipping audit log for {event_type} - organization_id not provided "
                f"(file: {file_name}, details: {details})"
            )
            return

        # Auto-derive action from event_type if not provided
        if action is None:
            if event_type in ("document_loaded", "document_agent_query", "cache_hit", "generation_cache_hit"):
                action = "DOWNLOAD"
            elif event_type in ("summary_generated", "faqs_generated", "questions_generated", "content_generated", "generation_started", "generation_completed"):
                action = "CREATE"
            elif event_type in ("parse_started", "parse_completed"):
                action = "UPDATE"
            elif event_type == "error":
                action = "UPDATE"
            else:
                action = "UPDATE"

        # Default entity_type to DOCUMENT for AI processing events
        if entity_type is None:
            entity_type = "DOCUMENT"

        # Default entity_id to file_name or a fallback
        # Note: entity_id is VARCHAR(36), document_hash is 64 chars (SHA-256)
        if entity_id is None:
            entity_id = file_name or "unknown"
            # Truncate to 36 chars if still too long
            if len(entity_id) > 36:
                entity_id = entity_id[:36]

        entry = AuditLog(
            organization_id=organization_id,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            created_at=datetime.utcnow(),
            event_type=event_type,
            document_hash=document_hash,
            file_name=file_name,
            job_id=job_id,
            details=details or {},
        )
        session.add(entry)

        # Explicitly flush to catch constraint errors early
        try:
            await session.flush()
        except IntegrityError as e:
            # Only IntegrityError indicates constraint violations
            error_str = str(e.orig).lower() if hasattr(e, 'orig') else str(e).lower()

            # Check for actual FK constraint violation patterns
            is_fk_violation = (
                "foreign key" in error_str or
                "fk_" in error_str or
                "violates foreign key constraint" in error_str
            )

            if is_fk_violation:
                logger.warning(
                    f"Skipping audit log for {event_type} - organization_id '{organization_id}' "
                    f"not found in organizations table (file: {file_name})"
                )
                await session.rollback()
                return

            # Log and re-raise other integrity errors
            logger.error(f"Integrity error logging audit event {event_type}: {e}")
            await session.rollback()
            raise
        except Exception as e:
            # Log unexpected errors with full details
            logger.error(f"Unexpected error logging audit event {event_type}: {e}")
            await session.rollback()
            raise


# =============================================================================
# AUDIT TRAIL QUERIES
# =============================================================================


@with_db_retry
async def get_audit_trail(
    file_name: Optional[str] = None,
    organization_id: Optional[str] = None,
    event_type: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """
    Get audit trail with flexible filtering and pagination.

    Multi-tenancy: Filtered by organization_id when provided.

    Args:
        file_name: Optional filter by document file name
        organization_id: Organization ID for tenant isolation
        event_type: Optional filter by event type
        start_date: Optional filter for events after this date
        end_date: Optional filter for events before this date
        limit: Maximum number of results
        offset: Number of results to skip for pagination

    Returns:
        List of audit events ordered by timestamp (newest first)
    """
    async with db.session() as session:
        where_clauses = []
        if file_name:
            where_clauses.append(AuditLog.file_name == file_name)
        if organization_id:
            where_clauses.append(AuditLog.organization_id == organization_id)
        if event_type:
            where_clauses.append(AuditLog.event_type == event_type)
        if start_date:
            where_clauses.append(AuditLog.created_at >= start_date)
        if end_date:
            where_clauses.append(AuditLog.created_at <= end_date)

        # Always filter out NULL event_type records (incomplete/legacy data)
        where_clauses.append(AuditLog.event_type.isnot(None))

        stmt = select(AuditLog)
        if where_clauses:
            stmt = stmt.where(and_(*where_clauses))
        stmt = stmt.order_by(desc(AuditLog.created_at)).offset(offset).limit(limit)

        result = await session.execute(stmt)
        events = result.scalars().all()

        return [
            {
                "id": event.id,
                "organization_id": event.organization_id,
                "created_at": event.created_at,
                "event_type": event.event_type,
                "document_hash": event.document_hash,
                "file_name": event.file_name,
                "job_id": event.job_id,
                "details": event.details,
            }
            for event in events
        ]


async def get_document_summary(
    file_name: str,
    organization_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get complete summary of a document including jobs and audit trail.

    Uses asyncio.gather for parallel queries.
    Multi-tenancy: Jobs and audit trail are filtered by organization_id when provided.

    Args:
        file_name: Name of the document file
        organization_id: Organization ID for tenant isolation

    Returns:
        Dictionary with document info, jobs, and audit trail
    """
    doc = await get_document_by_name(file_name)
    if not doc:
        return None

    # Run queries concurrently with org_id filtering
    jobs, audit_trail = await asyncio.gather(
        get_jobs_by_document(file_name, organization_id=organization_id),
        get_audit_trail(file_name, organization_id=organization_id),
    )

    return {
        "document": doc,
        "jobs": jobs,
        "audit_trail": audit_trail,
    }
