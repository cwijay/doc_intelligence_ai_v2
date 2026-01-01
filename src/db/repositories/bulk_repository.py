"""Bulk processing repository - Bulk job and document CRUD operations.

Multi-tenancy: All operations are scoped by organization_id for tenant isolation.
"""

import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy import select, and_, desc, func, update

from ..models import BulkJob, BulkJobDocument
from ..connection import db
from ..utils import with_db_retry

logger = logging.getLogger(__name__)


# =============================================================================
# BULK JOB CRUD
# =============================================================================


@with_db_retry
async def create_bulk_job(
    organization_id: str,
    folder_name: str,
    source_path: str,
    total_documents: int = 0,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a new bulk processing job.

    Args:
        organization_id: Organization ID for tenant isolation
        folder_name: Name of the bulk folder
        source_path: GCS path to source folder
        total_documents: Initial count of documents to process
        options: Processing options (generate_summary, generate_faqs, etc.)

    Returns:
        Created job as dictionary
    """
    async with db.session() as session:
        job = BulkJob(
            organization_id=organization_id,
            folder_name=folder_name,
            source_path=source_path,
            total_documents=total_documents,
            status="pending",
            options=options or {},
        )
        session.add(job)
        await session.flush()

        logger.info(f"Created bulk job {job.id} for folder '{folder_name}' org={organization_id}")
        return job.to_dict()


@with_db_retry
async def get_bulk_job(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a bulk job by ID.

    Args:
        job_id: Job ID to retrieve

    Returns:
        Job as dictionary or None if not found
    """
    async with db.session() as session:
        stmt = select(BulkJob).where(BulkJob.id == job_id)
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()
        return job.to_dict() if job else None


@with_db_retry
async def get_bulk_job_by_folder(
    organization_id: str,
    folder_name: str,
    status: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get a bulk job by folder name.

    Args:
        organization_id: Organization ID for tenant isolation
        folder_name: Name of the bulk folder
        status: Optional status filter

    Returns:
        Job as dictionary or None if not found
    """
    async with db.session() as session:
        where_clauses = [
            BulkJob.organization_id == organization_id,
            BulkJob.folder_name == folder_name,
        ]
        if status:
            where_clauses.append(BulkJob.status == status)

        stmt = (
            select(BulkJob)
            .where(and_(*where_clauses))
            .order_by(desc(BulkJob.created_at))
            .limit(1)
        )
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()
        return job.to_dict() if job else None


@with_db_retry
async def list_bulk_jobs(
    organization_id: str,
    status: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """
    List bulk jobs for an organization.

    Args:
        organization_id: Organization ID for tenant isolation
        status: Optional status filter
        limit: Maximum number of jobs to return
        offset: Number of jobs to skip

    Returns:
        List of jobs as dictionaries
    """
    async with db.session() as session:
        where_clauses = [BulkJob.organization_id == organization_id]
        if status:
            where_clauses.append(BulkJob.status == status)

        stmt = (
            select(BulkJob)
            .where(and_(*where_clauses))
            .order_by(desc(BulkJob.created_at))
            .limit(limit)
            .offset(offset)
        )
        result = await session.execute(stmt)
        jobs = result.scalars().all()
        return [job.to_dict() for job in jobs]


@with_db_retry
async def count_bulk_jobs(
    organization_id: str,
    status: Optional[str] = None,
) -> int:
    """
    Count bulk jobs for an organization.

    Args:
        organization_id: Organization ID for tenant isolation
        status: Optional status filter

    Returns:
        Count of matching jobs
    """
    async with db.session() as session:
        where_clauses = [BulkJob.organization_id == organization_id]
        if status:
            where_clauses.append(BulkJob.status == status)

        stmt = select(func.count(BulkJob.id)).where(and_(*where_clauses))
        result = await session.execute(stmt)
        return result.scalar() or 0


@with_db_retry
async def update_bulk_job_status(
    job_id: str,
    status: str,
    started_at: Optional[datetime] = None,
    completed_at: Optional[datetime] = None,
    error_message: Optional[str] = None,
) -> bool:
    """
    Update a bulk job's status.

    Args:
        job_id: Job ID to update
        status: New status value
        started_at: Optional start timestamp
        completed_at: Optional completion timestamp
        error_message: Optional error message

    Returns:
        True if job was updated, False if not found
    """
    async with db.session() as session:
        stmt = select(BulkJob).where(BulkJob.id == job_id)
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()

        if not job:
            return False

        job.status = status
        job.updated_at = datetime.utcnow()

        if started_at:
            job.started_at = started_at
        if completed_at:
            job.completed_at = completed_at
        if error_message is not None:
            job.error_message = error_message

        logger.debug(f"Updated bulk job {job_id} status to {status}")
        return True


@with_db_retry
async def increment_job_completed(
    job_id: str,
    token_usage: int = 0,
    llamaparse_pages: int = 0,
) -> bool:
    """
    Increment completed count and update totals for a bulk job.

    Args:
        job_id: Job ID to update
        token_usage: Tokens used by the completed document
        llamaparse_pages: LlamaParse pages used

    Returns:
        True if job was updated
    """
    async with db.session() as session:
        stmt = select(BulkJob).where(BulkJob.id == job_id)
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()

        if not job:
            return False

        job.completed_count += 1
        job.total_tokens_used += token_usage
        job.total_llamaparse_pages += llamaparse_pages
        job.updated_at = datetime.utcnow()

        logger.debug(f"Incremented completed count for job {job_id}: {job.completed_count}")
        return True


@with_db_retry
async def increment_job_failed(job_id: str) -> bool:
    """
    Increment failed count for a bulk job.

    Args:
        job_id: Job ID to update

    Returns:
        True if job was updated
    """
    async with db.session() as session:
        stmt = select(BulkJob).where(BulkJob.id == job_id)
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()

        if not job:
            return False

        job.failed_count += 1
        job.updated_at = datetime.utcnow()

        logger.debug(f"Incremented failed count for job {job_id}: {job.failed_count}")
        return True


@with_db_retry
async def increment_job_skipped(job_id: str) -> bool:
    """
    Increment skipped count for a bulk job.

    Args:
        job_id: Job ID to update

    Returns:
        True if job was updated
    """
    async with db.session() as session:
        stmt = select(BulkJob).where(BulkJob.id == job_id)
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()

        if not job:
            return False

        job.skipped_count += 1
        job.updated_at = datetime.utcnow()

        logger.debug(f"Incremented skipped count for job {job_id}: {job.skipped_count}")
        return True


@with_db_retry
async def update_job_document_count(job_id: str, total_documents: int) -> bool:
    """
    Update the total document count for a bulk job.

    Args:
        job_id: Job ID to update
        total_documents: New total document count

    Returns:
        True if job was updated
    """
    async with db.session() as session:
        stmt = select(BulkJob).where(BulkJob.id == job_id)
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()

        if not job:
            return False

        job.total_documents = total_documents
        job.updated_at = datetime.utcnow()
        return True


# =============================================================================
# BULK JOB DOCUMENT CRUD
# =============================================================================


@with_db_retry
async def create_document_item(
    bulk_job_id: str,
    original_path: str,
    original_filename: str,
) -> Dict[str, Any]:
    """
    Create a new document item within a bulk job.

    Args:
        bulk_job_id: Parent bulk job ID
        original_path: GCS path to original document
        original_filename: Original filename

    Returns:
        Created document item as dictionary
    """
    async with db.session() as session:
        doc = BulkJobDocument(
            bulk_job_id=bulk_job_id,
            original_path=original_path,
            original_filename=original_filename,
            status="pending",
        )
        session.add(doc)
        await session.flush()

        logger.debug(f"Created document item {doc.id} for job {bulk_job_id}: {original_filename}")
        return doc.to_dict()


@with_db_retry
async def get_document_item(document_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a document item by ID.

    Args:
        document_id: Document item ID

    Returns:
        Document item as dictionary or None
    """
    async with db.session() as session:
        stmt = select(BulkJobDocument).where(BulkJobDocument.id == document_id)
        result = await session.execute(stmt)
        doc = result.scalar_one_or_none()
        return doc.to_dict() if doc else None


@with_db_retry
async def get_document_by_path(
    bulk_job_id: str,
    original_path: str,
) -> Optional[Dict[str, Any]]:
    """
    Get a document item by its original path.

    Args:
        bulk_job_id: Parent bulk job ID
        original_path: GCS path to original document

    Returns:
        Document item as dictionary or None
    """
    async with db.session() as session:
        stmt = select(BulkJobDocument).where(
            and_(
                BulkJobDocument.bulk_job_id == bulk_job_id,
                BulkJobDocument.original_path == original_path,
            )
        )
        result = await session.execute(stmt)
        doc = result.scalar_one_or_none()
        return doc.to_dict() if doc else None


@with_db_retry
async def get_all_document_items(bulk_job_id: str) -> List[Dict[str, Any]]:
    """
    Get all document items for a bulk job.

    Args:
        bulk_job_id: Parent bulk job ID

    Returns:
        List of document items as dictionaries
    """
    async with db.session() as session:
        stmt = (
            select(BulkJobDocument)
            .where(BulkJobDocument.bulk_job_id == bulk_job_id)
            .order_by(BulkJobDocument.created_at)
        )
        result = await session.execute(stmt)
        docs = result.scalars().all()
        return [doc.to_dict() for doc in docs]


@with_db_retry
async def get_pending_documents(
    bulk_job_id: str,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Get pending document items for a bulk job.

    Args:
        bulk_job_id: Parent bulk job ID
        limit: Maximum number of documents to return

    Returns:
        List of pending document items
    """
    async with db.session() as session:
        stmt = (
            select(BulkJobDocument)
            .where(
                and_(
                    BulkJobDocument.bulk_job_id == bulk_job_id,
                    BulkJobDocument.status == "pending",
                )
            )
            .order_by(BulkJobDocument.created_at)
        )
        if limit:
            stmt = stmt.limit(limit)

        result = await session.execute(stmt)
        docs = result.scalars().all()
        return [doc.to_dict() for doc in docs]


@with_db_retry
async def get_failed_documents(bulk_job_id: str) -> List[Dict[str, Any]]:
    """
    Get failed document items for a bulk job.

    Args:
        bulk_job_id: Parent bulk job ID

    Returns:
        List of failed document items
    """
    async with db.session() as session:
        stmt = (
            select(BulkJobDocument)
            .where(
                and_(
                    BulkJobDocument.bulk_job_id == bulk_job_id,
                    BulkJobDocument.status == "failed",
                )
            )
            .order_by(BulkJobDocument.created_at)
        )
        result = await session.execute(stmt)
        docs = result.scalars().all()
        return [doc.to_dict() for doc in docs]


@with_db_retry
async def get_cancellable_documents(bulk_job_id: str) -> List[Dict[str, Any]]:
    """
    Get all documents that can be cancelled (pending or in-progress).

    Args:
        bulk_job_id: Parent bulk job ID

    Returns:
        List of cancellable document items
    """
    cancellable_statuses = ["pending", "parsing", "parsed", "indexing", "indexed", "generating"]

    async with db.session() as session:
        stmt = (
            select(BulkJobDocument)
            .where(
                and_(
                    BulkJobDocument.bulk_job_id == bulk_job_id,
                    BulkJobDocument.status.in_(cancellable_statuses),
                )
            )
            .order_by(BulkJobDocument.created_at)
        )
        result = await session.execute(stmt)
        docs = result.scalars().all()
        return [doc.to_dict() for doc in docs]


@with_db_retry
async def cancel_all_documents(bulk_job_id: str) -> int:
    """
    Cancel ALL cancellable documents in a job atomically.

    Uses a single SQL UPDATE statement to mark all pending/in-progress
    documents as SKIPPED. This prevents race conditions during cancellation.

    Args:
        bulk_job_id: Parent bulk job ID

    Returns:
        Count of documents cancelled
    """
    cancellable_statuses = ["pending", "parsing", "parsed", "indexing", "indexed", "generating"]

    async with db.session() as session:
        # Single UPDATE statement for all documents - atomic operation
        stmt = (
            update(BulkJobDocument)
            .where(
                and_(
                    BulkJobDocument.bulk_job_id == bulk_job_id,
                    BulkJobDocument.status.in_(cancellable_statuses),
                )
            )
            .values(
                status="skipped",
                error_message="Job cancelled",
                updated_at=datetime.utcnow(),
            )
        )
        result = await session.execute(stmt)
        await session.commit()

        cancelled_count = result.rowcount
        logger.info(f"Atomically cancelled {cancelled_count} documents for job {bulk_job_id}")
        return cancelled_count


@with_db_retry
async def set_job_skipped_count(bulk_job_id: str, count: int) -> bool:
    """
    Set the skipped count for a job (used during atomic cancellation).

    Args:
        bulk_job_id: Bulk job ID
        count: Number of skipped documents

    Returns:
        True if update was successful
    """
    async with db.session() as session:
        stmt = (
            update(BulkJob)
            .where(BulkJob.id == bulk_job_id)
            .values(skipped_count=count, updated_at=datetime.utcnow())
        )
        await session.execute(stmt)
        await session.commit()
        return True


@with_db_retry
async def count_in_progress_documents(bulk_job_id: str) -> int:
    """
    Count documents currently being processed.

    Args:
        bulk_job_id: Parent bulk job ID

    Returns:
        Count of in-progress documents
    """
    in_progress_statuses = ["parsing", "parsed", "indexing", "indexed", "generating"]

    async with db.session() as session:
        stmt = select(func.count(BulkJobDocument.id)).where(
            and_(
                BulkJobDocument.bulk_job_id == bulk_job_id,
                BulkJobDocument.status.in_(in_progress_statuses),
            )
        )
        result = await session.execute(stmt)
        return result.scalar() or 0


@with_db_retry
async def update_document_item(
    document_id: str,
    status: Optional[str] = None,
    parsed_path: Optional[str] = None,
    error_message: Optional[str] = None,
    parse_time_ms: Optional[int] = None,
    index_time_ms: Optional[int] = None,
    generation_time_ms: Optional[int] = None,
    total_time_ms: Optional[int] = None,
    token_usage: Optional[int] = None,
    llamaparse_pages: Optional[int] = None,
    content_hash: Optional[str] = None,
    retry_count: Optional[int] = None,
) -> bool:
    """
    Update a document item's fields.

    Args:
        document_id: Document item ID
        status: New status value
        parsed_path: Path to parsed document
        error_message: Error message if failed
        parse_time_ms: Parsing time in milliseconds
        index_time_ms: Indexing time in milliseconds
        generation_time_ms: Content generation time
        total_time_ms: Total processing time
        token_usage: Tokens used
        llamaparse_pages: LlamaParse pages used
        content_hash: SHA-256 hash of content
        retry_count: Number of retry attempts

    Returns:
        True if document was updated
    """
    async with db.session() as session:
        stmt = select(BulkJobDocument).where(BulkJobDocument.id == document_id)
        result = await session.execute(stmt)
        doc = result.scalar_one_or_none()

        if not doc:
            return False

        if status is not None:
            doc.status = status
        if parsed_path is not None:
            doc.parsed_path = parsed_path
        if error_message is not None:
            doc.error_message = error_message
        if parse_time_ms is not None:
            doc.parse_time_ms = parse_time_ms
        if index_time_ms is not None:
            doc.index_time_ms = index_time_ms
        if generation_time_ms is not None:
            doc.generation_time_ms = generation_time_ms
        if total_time_ms is not None:
            doc.total_time_ms = total_time_ms
        if token_usage is not None:
            doc.token_usage = token_usage
        if llamaparse_pages is not None:
            doc.llamaparse_pages = llamaparse_pages
        if content_hash is not None:
            doc.content_hash = content_hash
        if retry_count is not None:
            doc.retry_count = retry_count

        doc.updated_at = datetime.utcnow()

        logger.debug(f"Updated document item {document_id}: status={status}")
        return True


@with_db_retry
async def reset_document_for_retry(document_id: str) -> bool:
    """
    Reset a document item for retry processing.

    Args:
        document_id: Document item ID

    Returns:
        True if document was reset
    """
    async with db.session() as session:
        stmt = select(BulkJobDocument).where(BulkJobDocument.id == document_id)
        result = await session.execute(stmt)
        doc = result.scalar_one_or_none()

        if not doc:
            return False

        doc.status = "pending"
        doc.error_message = None
        doc.retry_count += 1
        doc.updated_at = datetime.utcnow()

        logger.debug(f"Reset document {document_id} for retry (attempt {doc.retry_count})")
        return True


@with_db_retry
async def bulk_reset_failed_documents(bulk_job_id: str) -> int:
    """
    Reset all failed documents in a bulk job for retry.

    Args:
        bulk_job_id: Parent bulk job ID

    Returns:
        Number of documents reset
    """
    async with db.session() as session:
        stmt = (
            update(BulkJobDocument)
            .where(
                and_(
                    BulkJobDocument.bulk_job_id == bulk_job_id,
                    BulkJobDocument.status == "failed",
                )
            )
            .values(
                status="pending",
                error_message=None,
                retry_count=BulkJobDocument.retry_count + 1,
                updated_at=datetime.utcnow(),
            )
        )
        result = await session.execute(stmt)
        count = result.rowcount

        # Also reset job failed count
        job_stmt = select(BulkJob).where(BulkJob.id == bulk_job_id)
        job_result = await session.execute(job_stmt)
        job = job_result.scalar_one_or_none()
        if job:
            job.failed_count = 0
            job.status = "processing"
            job.updated_at = datetime.utcnow()

        logger.info(f"Reset {count} failed documents in job {bulk_job_id}")
        return count


# =============================================================================
# CLEANUP AND MAINTENANCE
# =============================================================================


@with_db_retry
async def delete_bulk_job(job_id: str) -> bool:
    """
    Delete a bulk job and all its document items.

    Args:
        job_id: Job ID to delete

    Returns:
        True if job was deleted
    """
    async with db.session() as session:
        stmt = select(BulkJob).where(BulkJob.id == job_id)
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()

        if not job:
            return False

        await session.delete(job)  # Cascade deletes document items
        logger.info(f"Deleted bulk job {job_id}")
        return True


@with_db_retry
async def get_active_jobs_for_folder(
    organization_id: str,
    folder_name: str,
) -> List[Dict[str, Any]]:
    """
    Get all active (pending or processing) jobs for a folder.

    Args:
        organization_id: Organization ID for tenant isolation
        folder_name: Name of the bulk folder

    Returns:
        List of active jobs
    """
    async with db.session() as session:
        stmt = (
            select(BulkJob)
            .where(
                and_(
                    BulkJob.organization_id == organization_id,
                    BulkJob.folder_name == folder_name,
                    BulkJob.status.in_(["pending", "processing"]),
                )
            )
            .order_by(desc(BulkJob.created_at))
        )
        result = await session.execute(stmt)
        jobs = result.scalars().all()
        return [job.to_dict() for job in jobs]


@with_db_retry
async def find_active_job_for_folder(
    organization_id: str,
    folder_name: str,
) -> Optional[Dict[str, Any]]:
    """
    Find a single active (pending or processing) job for a folder.

    Args:
        organization_id: Organization ID for tenant isolation
        folder_name: Name of the bulk folder

    Returns:
        Most recent active job or None
    """
    async with db.session() as session:
        stmt = (
            select(BulkJob)
            .where(
                and_(
                    BulkJob.organization_id == organization_id,
                    BulkJob.folder_name == folder_name,
                    BulkJob.status.in_(["pending", "processing"]),
                )
            )
            .order_by(desc(BulkJob.created_at))
            .limit(1)
        )
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()
        return job.to_dict() if job else None


@with_db_retry
async def count_documents_in_job(bulk_job_id: str) -> int:
    """
    Count total documents in a bulk job.

    Args:
        bulk_job_id: Parent bulk job ID

    Returns:
        Count of documents
    """
    async with db.session() as session:
        stmt = select(func.count(BulkJobDocument.id)).where(
            BulkJobDocument.bulk_job_id == bulk_job_id
        )
        result = await session.execute(stmt)
        return result.scalar() or 0


@with_db_retry
async def increment_total_documents(job_id: str) -> bool:
    """
    Increment total document count for a bulk job.

    Args:
        job_id: Job ID to update

    Returns:
        True if job was updated
    """
    async with db.session() as session:
        stmt = select(BulkJob).where(BulkJob.id == job_id)
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()

        if not job:
            return False

        job.total_documents += 1
        job.updated_at = datetime.utcnow()
        return True


@with_db_retry
async def get_latest_document_in_job(bulk_job_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the most recently created document in a bulk job.

    Args:
        bulk_job_id: Parent bulk job ID

    Returns:
        Most recent document or None
    """
    async with db.session() as session:
        stmt = (
            select(BulkJobDocument)
            .where(BulkJobDocument.bulk_job_id == bulk_job_id)
            .order_by(desc(BulkJobDocument.created_at))
            .limit(1)
        )
        result = await session.execute(stmt)
        doc = result.scalar_one_or_none()
        return doc.to_dict() if doc else None


@with_db_retry
async def get_document_item_by_path(
    bulk_job_id: str,
    original_path: str,
) -> Optional[Dict[str, Any]]:
    """
    Get a document item by its original path.
    Alias for get_document_by_path for webhook handler compatibility.

    Args:
        bulk_job_id: Parent bulk job ID
        original_path: GCS path to original document

    Returns:
        Document item as dictionary or None
    """
    return await get_document_by_path(bulk_job_id, original_path)
