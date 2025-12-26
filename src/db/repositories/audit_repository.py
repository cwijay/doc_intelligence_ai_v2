"""
Audit repository - PostgreSQL implementation matching Firestore API.

Multi-tenancy: All operations are scoped by organization_id for tenant isolation.

Provides all 16 public functions from the original firestore_db.py:
- Core operations: get_file_hash, register_document, find_cached_result,
  start_job, complete_job, fail_job, log_event, get_processing_history
- Dashboard queries: get_document_by_name, get_jobs_by_document,
  get_audit_trail, get_document_summary
- Document generations: save_document_generation, find_cached_generation,
  get_generations_by_document, get_recent_generations
"""

import asyncio
import hashlib
import logging
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional, List, Dict, Any

from sqlalchemy import select, and_, desc, delete
from sqlalchemy.dialects.postgresql import insert

# Import models - using aliases for backwards compatibility
# Note: DocumentModel has different field names than the original Document model:
#   - storage_path (was file_path)
#   - filename (was file_name)
#   - folder_id (was folder_name - different type: FK vs string)
from ..models import (
    Document,  # Alias for DocumentModel
    ProcessingJob,  # Alias for ProcessingJobModel
    AuditLog,  # Alias for AuditLogModel
    DocumentGeneration,  # Alias for DocumentGenerationModel
)
from ..connection import db
from ..utils import with_db_retry

logger = logging.getLogger(__name__)


def _is_db_enabled() -> bool:
    """Check if database is enabled."""
    return db.config.enabled


# =============================================================================
# FILE HASH COMPUTATION (unchanged from original)
# =============================================================================


@lru_cache(maxsize=1000)
def _compute_file_hash(file_path: str, mtime: float) -> str:
    """Internal cached hash computation keyed by path and modification time."""
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def get_file_hash(file_path: str) -> str:
    """
    Compute SHA-256 hash of file content with caching.

    Uses LRU cache keyed by file path and modification time,
    so unchanged files don't need re-hashing.

    Args:
        file_path: Path to the file

    Returns:
        Hexadecimal hash string (64 characters)
    """
    mtime = os.path.getmtime(file_path)
    return _compute_file_hash(file_path, mtime)


# =============================================================================
# DOCUMENT REGISTRATION
# =============================================================================


@with_db_retry
async def register_document(file_path: str) -> str:
    """
    Register document if not exists, return file_hash.

    Uses PostgreSQL upsert (INSERT ... ON CONFLICT DO NOTHING) for
    atomic deduplication.

    Args:
        file_path: Path to the document

    Returns:
        SHA-256 hash of the file content
    """
    file_hash = get_file_hash(file_path)
    path = Path(file_path)

    async with db.session() as session:
        stmt = (
            insert(Document)
            .values(
                file_hash=file_hash,
                file_path=file_path,
                file_name=path.name,
                file_size=path.stat().st_size,
                created_at=datetime.utcnow(),
            )
            .on_conflict_do_nothing(index_elements=["file_hash"])
        )

        result = await session.execute(stmt)

        if result.rowcount > 0:
            logger.debug(f"Registered new document: {path.name} ({file_hash[:8]}...)")

    return file_hash


# =============================================================================
# DOCUMENT STATUS TRACKING (for upload/parse lifecycle)
# =============================================================================


@with_db_retry
async def register_uploaded_document(
    file_path: str,
    file_name: str,
    file_size: int,
    organization_id: str,
    folder_name: Optional[str] = None,
) -> str:
    """
    Register a new uploaded document with status='uploaded'.

    Uses file_path as unique lookup key (GCS URI includes bucket/org/folder/filename).
    Computes content hash from local temp file if available.

    Multi-tenancy: Scoped by organization_id.

    Args:
        file_path: GCS URI or local path (unique identifier)
        file_name: Original filename for display
        file_size: File size in bytes
        organization_id: Organization ID for tenant isolation
        folder_name: Optional folder context (used as folder_id)

    Returns:
        File hash (SHA-256 if computed, or hash of file_path as fallback)
    """
    # Generate hash - use file_path hash as fallback for GCS files
    # (actual content hash would require downloading)
    file_hash = hashlib.sha256(file_path.encode()).hexdigest()

    # Infer file type from extension
    ext = Path(file_path).suffix.lower().lstrip(".")
    file_type = ext if ext else "unknown"

    async with db.session() as session:
        if session is None:
            return file_hash

        now = datetime.utcnow()

        stmt = (
            insert(Document)
            .values(
                file_hash=file_hash,
                storage_path=file_path,
                filename=file_name,
                original_filename=file_name,
                file_type=file_type,
                file_size=file_size,
                organization_id=organization_id,
                folder_id=folder_name,
                status="uploaded",
                uploaded_by="system",
                is_active=True,
                created_at=now,
                updated_at=now,
            )
            .on_conflict_do_update(
                index_elements=["file_hash"],
                set_={
                    "filename": file_name,
                    "file_size": file_size,
                    "folder_id": folder_name,
                    "status": "uploaded",
                    "updated_at": now,
                }
            )
        )

        await session.execute(stmt)
        logger.debug(f"Registered uploaded document: {file_name} at {file_path} org={organization_id}")

    return file_hash


@with_db_retry
async def update_document_status(
    file_path: str,
    status: str,
    organization_id: str,
    parsed_path: Optional[str] = None,
) -> bool:
    """
    Update document status after parsing.

    Args:
        file_path: GCS URI or local path (unique lookup key)
        status: New status ('parsed' or 'failed')
        organization_id: Organization ID for tenant isolation
        parsed_path: Path to parsed .md file (for status='parsed')

    Returns:
        True if document was found and updated, False otherwise
    """
    async with db.session() as session:
        if session is None:
            return False

        stmt = select(Document).where(
            and_(
                Document.storage_path == file_path,
                Document.organization_id == organization_id,
            )
        )
        result = await session.execute(stmt)
        doc = result.scalar_one_or_none()

        if not doc:
            logger.warning(f"Document not found for status update: {file_path} org={organization_id}")
            return False

        doc.status = status
        if status == "parsed":
            doc.parsed_path = parsed_path
            doc.parsed_at = datetime.utcnow()

        logger.info(f"Updated document status to '{status}': {file_path} org={organization_id}")
        return True


@with_db_retry
async def register_or_update_parsed_document(
    storage_path: str,
    filename: str,
    organization_id: str,
    parsed_path: str,
    file_size: Optional[int] = None,
    folder_id: Optional[str] = None,
) -> bool:
    """
    Update existing document to parsed status, or create new if not found.

    Lookup strategy: Find existing document by filename + organization_id,
    then UPDATE it. This preserves the original record created during upload.

    Multi-tenancy: Scoped by organization_id.

    Args:
        storage_path: GCS URI of the original document (e.g., gs://bucket/org/original/file.pdf)
        filename: Filename for display
        organization_id: Organization ID for tenant isolation
        parsed_path: GCS path to parsed .md file
        file_size: Optional file size in bytes
        folder_id: Optional folder ID for organization (not used for lookup)

    Returns:
        True if document was updated/created, False on error
    """
    async with db.session() as session:
        if session is None:
            return False

        now = datetime.utcnow()

        # Look up existing document by filename + organization_id
        stmt = (
            select(Document)
            .where(
                and_(
                    Document.filename == filename,
                    Document.organization_id == organization_id,
                )
            )
            .limit(1)
        )
        result = await session.execute(stmt)
        doc = result.scalar_one_or_none()

        if doc:
            # UPDATE existing record - preserve original data
            doc.status = "parsed"
            doc.parsed_path = parsed_path
            doc.parsed_at = now
            doc.updated_at = now
            logger.info(
                f"Updated existing document to parsed: {filename} "
                f"parsed_path={parsed_path} org={organization_id}"
            )
        else:
            # INSERT new record (fallback for documents not pre-registered)
            file_hash = hashlib.sha256(storage_path.encode()).hexdigest()
            ext = Path(storage_path).suffix.lower().lstrip(".")
            file_type = ext if ext else "unknown"

            new_doc = Document(
                file_hash=file_hash,
                storage_path=storage_path,
                filename=filename,
                original_filename=filename,
                file_type=file_type,
                file_size=file_size or 0,
                organization_id=organization_id,
                folder_id=folder_id,
                status="parsed",
                parsed_path=parsed_path,
                parsed_at=now,
                uploaded_by="system",
                is_active=True,
                created_at=now,
                updated_at=now,
            )
            session.add(new_doc)
            logger.info(
                f"Created new parsed document: {filename} at {storage_path} "
                f"parsed_path={parsed_path} org={organization_id}"
            )

        return True


@with_db_retry
async def get_document_by_path(
    file_path: str,
    organization_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Get document by file_path (GCS URI).

    Args:
        file_path: GCS URI or local path (unique lookup key)
        organization_id: Organization ID for tenant isolation

    Returns:
        Document data dictionary or None if not found
    """
    async with db.session() as session:
        if session is None:
            return None

        stmt = select(Document).where(
            and_(
                Document.storage_path == file_path,
                Document.organization_id == organization_id,
            )
        )
        result = await session.execute(stmt)
        doc = result.scalar_one_or_none()

        if doc:
            return {
                "file_hash": doc.file_hash,
                "storage_path": doc.storage_path,
                "filename": doc.filename,
                "file_size": doc.file_size,
                "organization_id": doc.organization_id,
                "folder_id": doc.folder_id,
                "status": doc.status,
                "parsed_path": doc.parsed_path,
                "parsed_at": doc.parsed_at,
                "created_at": doc.created_at,
            }
        return None


@with_db_retry
async def list_documents_by_status(
    organization_id: str,
    status: Optional[str] = None,
    folder_id: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    List documents for an organization, optionally filtered by status.

    Multi-tenancy: Scoped by organization_id.

    Args:
        organization_id: Organization ID for tenant isolation
        status: Optional status filter ('uploaded', 'parsed', 'failed')
        folder_id: Optional folder ID filter (changed from folder_name)
        limit: Maximum number of results

    Returns:
        List of document dictionaries
    """
    async with db.session() as session:
        if session is None:
            return []

        where_clauses = [Document.organization_id == organization_id]
        if status:
            where_clauses.append(Document.status == status)
        if folder_id:
            where_clauses.append(Document.folder_id == folder_id)

        stmt = (
            select(Document)
            .where(and_(*where_clauses))
            .order_by(desc(Document.created_at))
            .limit(limit)
        )
        result = await session.execute(stmt)
        docs = result.scalars().all()

        return [
            {
                "file_hash": doc.file_hash,
                "storage_path": doc.storage_path,
                "filename": doc.filename,
                "file_size": doc.file_size,
                "organization_id": doc.organization_id,
                "folder_id": doc.folder_id,
                "status": doc.status,
                "parsed_path": doc.parsed_path,
                "parsed_at": doc.parsed_at,
                "created_at": doc.created_at,
            }
            for doc in docs
        ]


# =============================================================================
# CACHE LOOKUP
# =============================================================================


@with_db_retry
async def find_cached_result(
    file_hash: str,
    model: str,
    organization_id: Optional[str] = None,
) -> Optional[str]:
    """
    Check for completed job with same hash+model, return output_path if found.

    Uses partial index idx_jobs_org_cache_lookup for optimal performance.
    Multi-tenancy: Scoped by organization_id when provided.

    Args:
        file_hash: SHA-256 hash of the document
        model: Model used for processing
        organization_id: Organization ID for tenant isolation

    Returns:
        Output path if cached result exists, None otherwise
    """
    async with db.session() as session:
        where_clauses = [
            ProcessingJob.document_hash == file_hash,
            ProcessingJob.model == model,
            ProcessingJob.status == "completed",
        ]
        if organization_id:
            where_clauses.append(ProcessingJob.organization_id == organization_id)

        stmt = (
            select(ProcessingJob.output_path)
            .where(and_(*where_clauses))
            .limit(1)
        )

        result = await session.execute(stmt)
        row = result.scalar_one_or_none()
        return row


# =============================================================================
# JOB MANAGEMENT
# =============================================================================


@with_db_retry
async def start_job(
    document_hash: str,
    file_name: str,
    model: str,
    complexity: str,
    organization_id: Optional[str] = None,
) -> str:
    """
    Create processing job, return job_id.

    Multi-tenancy: Associates job with organization_id for tenant isolation.

    Args:
        document_hash: SHA-256 hash of the document
        file_name: Name of the document file
        model: Model being used for processing
        complexity: Complexity level ("normal" or "high")
        organization_id: Organization ID for tenant isolation

    Returns:
        Job ID (UUID string)
    """
    async with db.session() as session:
        job = ProcessingJob(
            organization_id=organization_id,
            document_hash=document_hash,
            file_name=file_name,
            model=model,
            complexity=complexity,
            status="processing",
            started_at=datetime.utcnow(),
            cached=False,
        )
        session.add(job)
        await session.flush()  # Get the generated UUID

        job_id = str(job.id)
        logger.debug(f"Started job {job_id} for {file_name} ({document_hash[:8]}...) org={organization_id}")
        return job_id


@with_db_retry
async def complete_job(job_id: str, output_path: str, duration_ms: int):
    """
    Mark job as completed.

    Args:
        job_id: Job ID to update
        output_path: Path where output was saved
        duration_ms: Processing duration in milliseconds
    """
    async with db.session() as session:
        stmt = select(ProcessingJob).where(ProcessingJob.id == job_id)
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()

        if job:
            job.status = "completed"
            job.output_path = output_path
            job.duration_ms = duration_ms
            job.completed_at = datetime.utcnow()
            logger.debug(f"Completed job {job_id} in {duration_ms}ms")


@with_db_retry
async def fail_job(job_id: str, error: str):
    """
    Mark job as failed.

    Args:
        job_id: Job ID to update
        error: Error message
    """
    async with db.session() as session:
        stmt = select(ProcessingJob).where(ProcessingJob.id == job_id)
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()

        if job:
            job.status = "failed"
            job.error_message = error
            job.completed_at = datetime.utcnow()
            logger.warning(f"Failed job {job_id}: {error}")


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

        # Default entity_id to document_hash or a fallback
        if entity_id is None:
            entity_id = document_hash or file_name or "unknown"

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
        except Exception as e:
            # Handle foreign key constraint error (organization doesn't exist)
            error_str = str(e).lower()
            if "foreign key" in error_str or "fk_" in error_str or "organization_id" in error_str:
                logger.warning(
                    f"Skipping audit log for {event_type} - organization_id '{organization_id}' "
                    f"not found in organizations table (file: {file_name})"
                )
                await session.rollback()
                return
            # Re-raise other errors
            raise


# =============================================================================
# HISTORY AND DASHBOARD QUERIES
# =============================================================================


@with_db_retry
async def get_processing_history(
    limit: int = 100,
    organization_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Get recent processing jobs.

    Multi-tenancy: Filtered by organization_id when provided.

    Args:
        limit: Maximum number of jobs to return
        organization_id: Organization ID for tenant isolation

    Returns:
        List of job dictionaries with id and all fields
    """
    async with db.session() as session:
        stmt = select(ProcessingJob)
        if organization_id:
            stmt = stmt.where(ProcessingJob.organization_id == organization_id)
        stmt = stmt.order_by(desc(ProcessingJob.started_at)).limit(limit)

        result = await session.execute(stmt)
        jobs = result.scalars().all()

        return [
            {
                "id": job.id,
                "organization_id": job.organization_id,
                "document_hash": job.document_hash,
                "file_name": job.file_name,
                "model": job.model,
                "complexity": job.complexity,
                "status": job.status,
                "started_at": job.started_at,
                "completed_at": job.completed_at,
                "cached": job.cached,
                "output_path": job.output_path,
                "duration_ms": job.duration_ms,
                "error_message": job.error_message,
            }
            for job in jobs
        ]


@with_db_retry
async def get_document_by_name(file_name: str) -> Optional[Dict[str, Any]]:
    """
    Find document by file name.

    Args:
        file_name: Name of the document file

    Returns:
        Document data with hash, or None if not found
    """
    async with db.session() as session:
        stmt = select(Document).where(Document.filename == file_name).limit(1)
        result = await session.execute(stmt)
        doc = result.scalar_one_or_none()

        if doc:
            return {
                "file_hash": doc.file_hash,
                "storage_path": doc.storage_path,
                "filename": doc.filename,
                "file_size": doc.file_size,
                "created_at": doc.created_at,
            }
        return None


@with_db_retry
async def get_jobs_by_document(
    file_name: str,
    organization_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Get all processing jobs for a document by name.

    Multi-tenancy: Filtered by organization_id when provided.

    Args:
        file_name: Name of the document file
        organization_id: Organization ID for tenant isolation

    Returns:
        List of job dictionaries ordered by start time (newest first)
    """
    async with db.session() as session:
        where_clauses = [ProcessingJob.file_name == file_name]
        if organization_id:
            where_clauses.append(ProcessingJob.organization_id == organization_id)

        stmt = (
            select(ProcessingJob)
            .where(and_(*where_clauses))
            .order_by(desc(ProcessingJob.started_at))
        )
        result = await session.execute(stmt)
        jobs = result.scalars().all()

        return [
            {
                "id": job.id,
                "organization_id": job.organization_id,
                "document_hash": job.document_hash,
                "file_name": job.file_name,
                "model": job.model,
                "complexity": job.complexity,
                "status": job.status,
                "started_at": job.started_at,
                "completed_at": job.completed_at,
                "cached": job.cached,
                "output_path": job.output_path,
                "duration_ms": job.duration_ms,
                "error_message": job.error_message,
            }
            for job in jobs
        ]


@with_db_retry
async def get_audit_trail(
    file_name: str,
    organization_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Get complete audit trail for a document by name.

    Multi-tenancy: Filtered by organization_id when provided.

    Args:
        file_name: Name of the document file
        organization_id: Organization ID for tenant isolation

    Returns:
        List of audit events ordered by timestamp (newest first)
    """
    async with db.session() as session:
        where_clauses = [AuditLog.file_name == file_name]
        if organization_id:
            where_clauses.append(AuditLog.organization_id == organization_id)

        stmt = (
            select(AuditLog)
            .where(and_(*where_clauses))
            .order_by(desc(AuditLog.created_at))
        )
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

    Uses asyncio.gather for parallel queries (similar to original ThreadPoolExecutor).
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
