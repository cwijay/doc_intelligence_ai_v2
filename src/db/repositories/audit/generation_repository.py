"""Generation repository - Document generation CRUD operations.

Multi-tenancy: All operations are scoped by organization_id for tenant isolation.
"""

import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy import select, and_, desc, delete

from ...models import Document, ProcessingJob, AuditLog, DocumentGeneration
from ...connection import db
from ...utils import with_db_retry

logger = logging.getLogger(__name__)


# =============================================================================
# DOCUMENT GENERATIONS
# =============================================================================


@with_db_retry
async def save_document_generation(
    document_name: str,
    source_path: str,
    generation_type: str,
    content: dict,
    options: dict,
    model: str,
    processing_time_ms: float,
    session_id: Optional[str] = None,
    document_hash: Optional[str] = None,
    organization_id: Optional[str] = None,
) -> Optional[str]:
    """
    Save generated content (summary, FAQs, questions) to PostgreSQL.

    Multi-tenancy: Associates generation with organization_id for tenant isolation.

    Args:
        document_name: Name of the source document
        source_path: Path where document was found
        generation_type: Type of generation ("summary", "faqs", "questions", "all")
        content: Generated content dict with summary, faqs, questions
        options: Generation options used
        model: LLM model used
        processing_time_ms: Processing duration
        session_id: Session ID if available
        document_hash: Document hash if available
        organization_id: Organization ID for tenant isolation

    Returns:
        PostgreSQL document ID (UUID string), or None if database disabled
    """
    async with db.session() as session:
        if session is None:
            # Database disabled - skip saving
            return None

        generation = DocumentGeneration(
            organization_id=organization_id,
            document_hash=document_hash,
            document_name=document_name,
            source_path=source_path,
            generation_type=generation_type,
            content=content,
            options=options,
            model=model,
            processing_time_ms=processing_time_ms,
            session_id=session_id,
            created_at=datetime.utcnow(),
        )
        session.add(generation)
        await session.flush()

        gen_id = str(generation.id)
        logger.debug(f"Saved document generation for {document_name}: {gen_id} org={organization_id}")
        return gen_id


@with_db_retry
async def find_cached_generation(
    document_name: str,
    generation_type: str,
    model: str,
    content_hash: Optional[str] = None,
    organization_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Find cached generation result for a document.

    Multi-tenancy: Filtered by organization_id when provided.

    Args:
        document_name: Name of the document
        generation_type: Type of generation
        model: Model used
        content_hash: SHA-256 hash of document content for cache validation.
                     If provided, only returns cached result if content unchanged.
        organization_id: Organization ID for tenant isolation

    Returns:
        Cached generation data if found and valid, None otherwise
    """
    async with db.session() as session:
        # Build where clauses
        where_clauses = [
            DocumentGeneration.document_name == document_name,
            DocumentGeneration.generation_type == generation_type,
            DocumentGeneration.model == model,
        ]

        # If content_hash provided, only return cache if document content unchanged
        if content_hash:
            where_clauses.append(DocumentGeneration.document_hash == content_hash)

        # Multi-tenancy filter
        if organization_id:
            where_clauses.append(DocumentGeneration.organization_id == organization_id)

        stmt = (
            select(DocumentGeneration)
            .where(and_(*where_clauses))
            .order_by(desc(DocumentGeneration.created_at))
            .limit(1)
        )
        result = await session.execute(stmt)
        gen = result.scalar_one_or_none()

        if gen:
            return {
                "id": gen.id,
                "organization_id": gen.organization_id,
                "document_hash": gen.document_hash,
                "document_name": gen.document_name,
                "source_path": gen.source_path,
                "generation_type": gen.generation_type,
                "content": gen.content,
                "options": gen.options,
                "model": gen.model,
                "processing_time_ms": gen.processing_time_ms,
                "session_id": gen.session_id,
                "created_at": gen.created_at,
            }
        return None


@with_db_retry
async def get_generations_by_document(
    document_name: str,
    limit: int = 10,
    organization_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Get all generations for a document.

    Multi-tenancy: Filtered by organization_id when provided.

    Args:
        document_name: Name of the document
        limit: Maximum number of results
        organization_id: Organization ID for tenant isolation

    Returns:
        List of generation records ordered by creation time (newest first)
    """
    async with db.session() as session:
        where_clauses = [DocumentGeneration.document_name == document_name]
        if organization_id:
            where_clauses.append(DocumentGeneration.organization_id == organization_id)

        stmt = (
            select(DocumentGeneration)
            .where(and_(*where_clauses))
            .order_by(desc(DocumentGeneration.created_at))
            .limit(limit)
        )
        result = await session.execute(stmt)
        generations = result.scalars().all()

        return [
            {
                "id": gen.id,
                "organization_id": gen.organization_id,
                "document_hash": gen.document_hash,
                "document_name": gen.document_name,
                "source_path": gen.source_path,
                "generation_type": gen.generation_type,
                "content": gen.content,
                "options": gen.options,
                "model": gen.model,
                "processing_time_ms": gen.processing_time_ms,
                "session_id": gen.session_id,
                "created_at": gen.created_at,
            }
            for gen in generations
        ]


@with_db_retry
async def get_recent_generations(
    limit: int = 50,
    organization_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Get recent document generations across all documents.

    Multi-tenancy: Filtered by organization_id when provided.

    Args:
        limit: Maximum number of results
        organization_id: Organization ID for tenant isolation

    Returns:
        List of generation records ordered by creation time (newest first)
    """
    async with db.session() as session:
        stmt = select(DocumentGeneration)
        if organization_id:
            stmt = stmt.where(DocumentGeneration.organization_id == organization_id)
        stmt = stmt.order_by(desc(DocumentGeneration.created_at)).limit(limit)

        result = await session.execute(stmt)
        generations = result.scalars().all()

        return [
            {
                "id": gen.id,
                "organization_id": gen.organization_id,
                "document_hash": gen.document_hash,
                "document_name": gen.document_name,
                "source_path": gen.source_path,
                "generation_type": gen.generation_type,
                "content": gen.content,
                "options": gen.options,
                "model": gen.model,
                "processing_time_ms": gen.processing_time_ms,
                "session_id": gen.session_id,
                "created_at": gen.created_at,
            }
            for gen in generations
        ]


# =============================================================================
# TEST DATA CLEANUP
# =============================================================================


@with_db_retry
async def delete_test_records(
    prefix: str = "test-",
    organization_id: Optional[str] = None,
) -> Dict[str, int]:
    """
    Delete all test records from database tables.

    Used for cleaning up after integration tests. Deletes records where
    file_name or document_name starts with the given prefix.
    Multi-tenancy: Filters by organization_id when provided.

    Args:
        prefix: Prefix to match for deletion (default: "test-")
        organization_id: Organization ID for tenant isolation

    Returns:
        Dictionary with count of deleted records per table
    """
    deleted_counts = {
        "audit_logs": 0,
        "processing_jobs": 0,
        "documents": 0,
        "document_generations": 0,
    }

    async with db.session() as session:
        if session is None:
            # Database disabled
            return deleted_counts

        # Delete from audit_logs first (no foreign key dependencies)
        where_clauses = [AuditLog.file_name.like(f"{prefix}%")]
        if organization_id:
            where_clauses.append(AuditLog.organization_id == organization_id)
        stmt = delete(AuditLog).where(and_(*where_clauses))
        result = await session.execute(stmt)
        deleted_counts["audit_logs"] = result.rowcount

        # Delete from processing_jobs
        where_clauses = [ProcessingJob.file_name.like(f"{prefix}%")]
        if organization_id:
            where_clauses.append(ProcessingJob.organization_id == organization_id)
        stmt = delete(ProcessingJob).where(and_(*where_clauses))
        result = await session.execute(stmt)
        deleted_counts["processing_jobs"] = result.rowcount

        # Delete from document_generations
        where_clauses = [DocumentGeneration.document_name.like(f"{prefix}%")]
        if organization_id:
            where_clauses.append(DocumentGeneration.organization_id == organization_id)
        stmt = delete(DocumentGeneration).where(and_(*where_clauses))
        result = await session.execute(stmt)
        deleted_counts["document_generations"] = result.rowcount

        # Delete from documents
        stmt = delete(Document).where(Document.filename.like(f"{prefix}%"))
        result = await session.execute(stmt)
        deleted_counts["documents"] = result.rowcount

        logger.info(
            f"Deleted test records with prefix '{prefix}' org={organization_id}: "
            f"audit_logs={deleted_counts['audit_logs']}, "
            f"processing_jobs={deleted_counts['processing_jobs']}, "
            f"document_generations={deleted_counts['document_generations']}, "
            f"documents={deleted_counts['documents']}"
        )

    return deleted_counts
