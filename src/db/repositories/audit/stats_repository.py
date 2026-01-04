"""Statistics repository - Aggregation queries for audit dashboard.

Multi-tenancy: All operations are scoped by organization_id for tenant isolation.
"""

import logging
from datetime import datetime
from time import time
from typing import Optional, Dict, Any

from sqlalchemy import select, and_, func, case

from ...models import Document, ProcessingJob, DocumentGeneration, AuditLog
from ...connection import db
from ...utils import with_db_retry

logger = logging.getLogger(__name__)


# =============================================================================
# CACHING
# =============================================================================

# Simple time-based cache for dashboard stats
_stats_cache: Dict[str, tuple[float, Dict[str, Any]]] = {}
STATS_CACHE_TTL_SECONDS = 30


def _get_cache_key(
    org_id: str,
    start_date: Optional[datetime],
    end_date: Optional[datetime],
) -> str:
    """Generate cache key from parameters."""
    start_str = start_date.isoformat() if start_date else "none"
    end_str = end_date.isoformat() if end_date else "none"
    return f"{org_id}:{start_str}:{end_str}"


def _empty_stats() -> Dict[str, Any]:
    """Return empty stats structure."""
    return {
        "total_documents": 0,
        "total_jobs": 0,
        "total_generations": 0,
        "jobs_by_status": {"completed": 0, "processing": 0, "failed": 0},
        "generations_by_type": {"summary": 0, "faqs": 0, "questions": 0, "all": 0},
        "cache_hit_rate": 0.0,
        "avg_processing_time_ms": 0.0,
    }


# =============================================================================
# DASHBOARD STATS
# =============================================================================


@with_db_retry
async def get_dashboard_stats(
    organization_id: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Get aggregated dashboard statistics for an organization.

    Args:
        organization_id: Organization ID for tenant isolation
        start_date: Optional filter for stats calculation period
        end_date: Optional filter for stats calculation period
        use_cache: Whether to use cached results (default True)

    Returns:
        Dictionary with dashboard statistics
    """
    # Check cache
    if use_cache:
        cache_key = _get_cache_key(organization_id, start_date, end_date)
        if cache_key in _stats_cache:
            cached_time, cached_data = _stats_cache[cache_key]
            if time() - cached_time < STATS_CACHE_TTL_SECONDS:
                logger.debug(f"Dashboard stats cache hit for org={organization_id}")
                return cached_data

    async with db.session() as session:
        if session is None:
            return _empty_stats()

        # Build date filters
        job_where = [ProcessingJob.organization_id == organization_id]
        gen_where = [DocumentGeneration.organization_id == organization_id]
        doc_where = [Document.organization_id == organization_id]

        if start_date:
            job_where.append(ProcessingJob.started_at >= start_date)
            gen_where.append(DocumentGeneration.created_at >= start_date)
        if end_date:
            job_where.append(ProcessingJob.started_at <= end_date)
            gen_where.append(DocumentGeneration.created_at <= end_date)

        # Total documents (no date filter - cumulative)
        doc_count_stmt = select(func.count(Document.id)).where(and_(*doc_where))
        total_documents = (await session.execute(doc_count_stmt)).scalar() or 0

        # Jobs with status breakdown and cache stats
        jobs_stmt = select(
            func.count(ProcessingJob.id).label("total"),
            func.sum(case((ProcessingJob.status == "completed", 1), else_=0)).label("completed"),
            func.sum(case((ProcessingJob.status == "processing", 1), else_=0)).label("processing"),
            func.sum(case((ProcessingJob.status == "failed", 1), else_=0)).label("failed"),
            func.sum(case((ProcessingJob.cached == True, 1), else_=0)).label("cached"),
            func.avg(ProcessingJob.duration_ms).label("avg_duration"),
        ).where(and_(*job_where))

        job_result = (await session.execute(jobs_stmt)).one()

        total_jobs = job_result.total or 0
        jobs_by_status = {
            "completed": int(job_result.completed or 0),
            "processing": int(job_result.processing or 0),
            "failed": int(job_result.failed or 0),
        }
        cached_jobs = int(job_result.cached or 0)
        avg_processing_time_ms = float(job_result.avg_duration or 0)

        # Cache hit rate = cached / total (where total > 0)
        cache_hit_rate = (cached_jobs / total_jobs) if total_jobs > 0 else 0.0

        # Generations by type
        gens_stmt = select(
            func.count(DocumentGeneration.id).label("total"),
            func.sum(case((DocumentGeneration.generation_type == "summary", 1), else_=0)).label("summary"),
            func.sum(case((DocumentGeneration.generation_type == "faqs", 1), else_=0)).label("faqs"),
            func.sum(case((DocumentGeneration.generation_type == "questions", 1), else_=0)).label("questions"),
            func.sum(case((DocumentGeneration.generation_type == "all", 1), else_=0)).label("all"),
        ).where(and_(*gen_where))

        gen_result = (await session.execute(gens_stmt)).one()

        total_generations = gen_result.total or 0
        generations_by_type = {
            "summary": int(gen_result.summary or 0),
            "faqs": int(gen_result.faqs or 0),
            "questions": int(gen_result.questions or 0),
            "all": int(gen_result.all or 0),
        }

        stats = {
            "total_documents": total_documents,
            "total_jobs": total_jobs,
            "total_generations": total_generations,
            "jobs_by_status": jobs_by_status,
            "generations_by_type": generations_by_type,
            "cache_hit_rate": round(cache_hit_rate, 4),
            "avg_processing_time_ms": round(avg_processing_time_ms, 2),
        }

        # Update cache
        if use_cache:
            cache_key = _get_cache_key(organization_id, start_date, end_date)
            _stats_cache[cache_key] = (time(), stats)

            # Cleanup old entries
            current_time = time()
            expired_keys = [
                k for k, (t, _) in _stats_cache.items()
                if current_time - t > STATS_CACHE_TTL_SECONDS * 2
            ]
            for k in expired_keys:
                del _stats_cache[k]

        return stats


# =============================================================================
# SINGLE ITEM LOOKUPS
# =============================================================================


@with_db_retry
async def get_job_by_id(
    job_id: str,
    organization_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get a single processing job by ID.

    Args:
        job_id: Job ID (UUID)
        organization_id: Optional org filter for multi-tenancy

    Returns:
        Job as dictionary or None if not found
    """
    async with db.session() as session:
        where_clauses = [ProcessingJob.id == job_id]
        if organization_id:
            where_clauses.append(ProcessingJob.organization_id == organization_id)

        stmt = select(ProcessingJob).where(and_(*where_clauses))
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()

        if not job:
            return None

        return {
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


# =============================================================================
# COUNT FUNCTIONS
# =============================================================================


@with_db_retry
async def count_jobs(
    organization_id: str,
    status: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> int:
    """Count jobs matching criteria."""
    async with db.session() as session:
        where_clauses = [ProcessingJob.organization_id == organization_id]
        if status:
            where_clauses.append(ProcessingJob.status == status)
        if start_date:
            where_clauses.append(ProcessingJob.started_at >= start_date)
        if end_date:
            where_clauses.append(ProcessingJob.started_at <= end_date)

        stmt = select(func.count(ProcessingJob.id)).where(and_(*where_clauses))
        result = await session.execute(stmt)
        return result.scalar() or 0


@with_db_retry
async def count_documents(
    organization_id: str,
    status: Optional[str] = None,
) -> int:
    """Count documents matching criteria."""
    async with db.session() as session:
        where_clauses = [Document.organization_id == organization_id]
        if status:
            where_clauses.append(Document.status == status)

        stmt = select(func.count(Document.id)).where(and_(*where_clauses))
        result = await session.execute(stmt)
        return result.scalar() or 0


@with_db_retry
async def count_generations(
    organization_id: str,
    generation_type: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> int:
    """Count generations matching criteria."""
    async with db.session() as session:
        where_clauses = [DocumentGeneration.organization_id == organization_id]
        if generation_type:
            where_clauses.append(DocumentGeneration.generation_type == generation_type)
        if start_date:
            where_clauses.append(DocumentGeneration.created_at >= start_date)
        if end_date:
            where_clauses.append(DocumentGeneration.created_at <= end_date)

        stmt = select(func.count(DocumentGeneration.id)).where(and_(*where_clauses))
        result = await session.execute(stmt)
        return result.scalar() or 0


@with_db_retry
async def count_audit_events(
    organization_id: str,
    event_type: Optional[str] = None,
    file_name: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> int:
    """Count audit events matching criteria."""
    async with db.session() as session:
        where_clauses = [AuditLog.organization_id == organization_id]
        if event_type:
            where_clauses.append(AuditLog.event_type == event_type)
        if file_name:
            where_clauses.append(AuditLog.file_name == file_name)
        if start_date:
            where_clauses.append(AuditLog.created_at >= start_date)
        if end_date:
            where_clauses.append(AuditLog.created_at <= end_date)

        stmt = select(func.count(AuditLog.id)).where(and_(*where_clauses))
        result = await session.execute(stmt)
        return result.scalar() or 0
