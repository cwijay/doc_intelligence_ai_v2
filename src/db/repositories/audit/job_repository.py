"""Job repository - Processing job lifecycle management.

Multi-tenancy: All operations are scoped by organization_id for tenant isolation.
"""

import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy import select, and_, desc

from ...models import ProcessingJob
from ...connection import db
from ...utils import with_db_retry

logger = logging.getLogger(__name__)


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
# JOB QUERIES
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
