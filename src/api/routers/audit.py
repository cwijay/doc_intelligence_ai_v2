"""Audit and Analytics API endpoints.

Multi-tenancy: All endpoints are scoped by organization_id from request headers.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from ..dependencies import get_org_id
from ..schemas.errors import BASE_ERROR_RESPONSES
from ..schemas.audit import (
    ListJobsResponse,
    GetJobResponse,
    ListDocumentsResponse,
    GetDocumentResponse,
    ListGenerationsResponse,
    AuditTrailResponse,
    LogEventRequest,
    LogEventResponse,
    ProcessingJob,
    DocumentRecord,
    DocumentGeneration,
    AuditEvent,
    DashboardStats,
    DashboardResponse,
    ActivityTimelineItem,
    ActivityTimelineResponse,
)
from ..utils.formatting import (
    format_time_ago,
    get_status_color,
    get_activity_title,
    get_activity_icon,
    build_activity_description,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# =============================================================================
# DASHBOARD ENDPOINT
# =============================================================================


@router.get(
    "/dashboard",
    response_model=DashboardResponse,
    responses=BASE_ERROR_RESPONSES,
    operation_id="getDashboard",
    summary="Get dashboard statistics",
)
async def get_dashboard(
    period: str = Query(
        default="30d",
        description="Stats period: 7d, 30d, 90d, or all",
        pattern="^(7d|30d|90d|all)$",
    ),
    org_id: str = Depends(get_org_id),
):
    """
    Get aggregated dashboard statistics for the organization.

    Returns:
    - Total counts for documents, jobs, generations
    - Status breakdown for jobs
    - Type breakdown for generations
    - Cache hit rate and average processing time
    - Recent jobs and generations

    **Multi-tenancy**: Scoped by X-Organization-ID header.
    """
    try:
        from src.db.repositories.audit import (
            get_dashboard_stats,
            get_processing_history,
            get_recent_generations,
        )

        # Calculate date range
        end_date = datetime.utcnow()
        start_date = None
        if period != "all":
            days = {"7d": 7, "30d": 30, "90d": 90}[period]
            start_date = end_date - timedelta(days=days)

        # Get stats
        stats_data = await get_dashboard_stats(
            organization_id=org_id,
            start_date=start_date,
            end_date=end_date,
        )

        # Get recent data
        recent_jobs_data = await get_processing_history(
            limit=5,
            organization_id=org_id,
        )

        recent_gens_data = await get_recent_generations(
            limit=5,
            organization_id=org_id,
        )

        # Build response
        stats = DashboardStats(
            total_documents=stats_data["total_documents"],
            total_jobs=stats_data["total_jobs"],
            total_generations=stats_data["total_generations"],
            jobs_by_status=stats_data["jobs_by_status"],
            generations_by_type=stats_data["generations_by_type"],
            cache_hit_rate=stats_data["cache_hit_rate"],
            avg_processing_time_ms=stats_data["avg_processing_time_ms"],
        )

        recent_jobs = [
            ProcessingJob(
                id=str(j["id"]),
                document_hash=j["document_hash"],
                file_name=j["file_name"],
                model=j["model"],
                complexity=j.get("complexity", "normal"),
                status=j["status"],
                started_at=j["started_at"],
                completed_at=j.get("completed_at"),
                cached=j.get("cached", False),
                output_path=j.get("output_path"),
                duration_ms=j.get("duration_ms"),
                error_message=j.get("error_message"),
            )
            for j in recent_jobs_data
        ]

        recent_generations = [
            DocumentGeneration(
                id=str(g["id"]),
                document_name=g["document_name"],
                document_hash=g.get("document_hash"),
                source_path=g.get("source_path"),
                generation_type=g["generation_type"],
                model=g["model"],
                processing_time_ms=g.get("processing_time_ms"),
                session_id=g.get("session_id"),
                created_at=g["created_at"],
            )
            for g in recent_gens_data
        ]

        return DashboardResponse(
            success=True,
            stats=stats,
            recent_jobs=recent_jobs,
            recent_generations=recent_generations,
        )

    except ImportError:
        return DashboardResponse(
            success=False,
            error="Audit repository not available",
        )
    except Exception as e:
        logger.exception(f"Failed to get dashboard: {e}")
        return DashboardResponse(
            success=False,
            error=str(e),
        )


# =============================================================================
# JOBS ENDPOINTS
# =============================================================================


@router.get(
    "/jobs",
    response_model=ListJobsResponse,
    responses=BASE_ERROR_RESPONSES,
    operation_id="listJobs",
    summary="List processing jobs",
)
async def list_jobs(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status: Optional[str] = Query(default=None, description="Filter by status: pending, processing, completed, failed"),
    start_date: Optional[datetime] = Query(default=None, description="Filter jobs started after this date (ISO 8601)"),
    end_date: Optional[datetime] = Query(default=None, description="Filter jobs started before this date (ISO 8601)"),
    org_id: str = Depends(get_org_id),
):
    """
    List recent document processing jobs for the organization.

    Supports filtering by status and date range, with pagination.

    **Multi-tenancy**: Scoped by X-Organization-ID header.
    """
    try:
        from src.db.repositories.audit import get_processing_history, count_jobs

        jobs = await get_processing_history(
            limit=limit,
            offset=offset,
            organization_id=org_id,
            status=status,
            start_date=start_date,
            end_date=end_date,
        )

        total = await count_jobs(
            organization_id=org_id,
            status=status,
            start_date=start_date,
            end_date=end_date,
        )

        result_jobs = [
            ProcessingJob(
                id=str(j["id"]),
                document_hash=j["document_hash"],
                file_name=j["file_name"],
                model=j["model"],
                complexity=j.get("complexity", "normal"),
                status=j["status"],
                started_at=j["started_at"],
                completed_at=j.get("completed_at"),
                cached=j.get("cached", False),
                output_path=j.get("output_path"),
                duration_ms=j.get("duration_ms"),
                error_message=j.get("error_message"),
            )
            for j in jobs
        ]

        return ListJobsResponse(
            success=True,
            jobs=result_jobs,
            total=total,
            limit=limit,
            offset=offset,
        )

    except ImportError:
        return ListJobsResponse(
            success=False,
            jobs=[],
            total=0,
            limit=limit,
            offset=offset,
            error="Audit repository not available"
        )
    except Exception as e:
        logger.exception(f"Failed to list jobs: {e}")
        return ListJobsResponse(
            success=False,
            jobs=[],
            total=0,
            limit=limit,
            offset=offset,
            error=str(e)
        )


@router.get(
    "/jobs/{job_id}",
    response_model=GetJobResponse,
    responses=BASE_ERROR_RESPONSES,
    operation_id="getJob",
    summary="Get job details",
)
async def get_job(
    job_id: str,
    org_id: str = Depends(get_org_id),
):
    """
    Get details of a specific processing job.

    **Multi-tenancy**: Scoped by X-Organization-ID header.
    """
    try:
        from src.db.repositories.audit import get_job_by_id

        job_data = await get_job_by_id(job_id=job_id, organization_id=org_id)

        if not job_data:
            return GetJobResponse(
                success=False,
                error=f"Job not found: {job_id}",
            )

        job = ProcessingJob(
            id=str(job_data["id"]),
            document_hash=job_data["document_hash"],
            file_name=job_data["file_name"],
            model=job_data["model"],
            complexity=job_data.get("complexity", "normal"),
            status=job_data["status"],
            started_at=job_data["started_at"],
            completed_at=job_data.get("completed_at"),
            cached=job_data.get("cached", False),
            output_path=job_data.get("output_path"),
            duration_ms=job_data.get("duration_ms"),
            error_message=job_data.get("error_message"),
        )

        return GetJobResponse(
            success=True,
            job=job,
        )

    except ImportError:
        return GetJobResponse(
            success=False,
            error="Audit repository not available",
        )
    except Exception as e:
        logger.exception(f"Failed to get job {job_id}: {e}")
        return GetJobResponse(
            success=False,
            error=str(e),
        )


@router.get(
    "/documents",
    response_model=ListDocumentsResponse,
    responses=BASE_ERROR_RESPONSES,
    operation_id="listDocuments",
    summary="List registered documents",
)
async def list_documents(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status: Optional[str] = Query(default=None, description="Filter by status: uploaded, parsed, failed"),
    folder_id: Optional[str] = Query(default=None, description="Filter by folder ID"),
    org_id: str = Depends(get_org_id),
):
    """
    List all registered documents for the organization.

    **Multi-tenancy**: Scoped by X-Organization-ID header.
    """
    try:
        from src.db.repositories.audit import list_documents_by_status, count_documents

        docs = await list_documents_by_status(
            organization_id=org_id,
            status=status,
            folder_id=folder_id,
            limit=limit + offset,  # Get enough to handle offset
        )

        # Apply offset manually (list_documents_by_status doesn't support offset)
        docs = docs[offset:offset + limit] if offset else docs[:limit]

        total = await count_documents(
            organization_id=org_id,
            status=status,
        )

        result_docs = [
            DocumentRecord(
                organization_id=d.get("organization_id"),
                file_hash=d.get("file_hash") or "",
                storage_path=d.get("storage_path") or "",
                filename=d.get("filename") or "",
                file_size=d.get("file_size") or 0,
                created_at=d.get("created_at"),
            )
            for d in docs
        ]

        return ListDocumentsResponse(
            success=True,
            documents=result_docs,
            total=total,
            limit=limit,
            offset=offset,
        )

    except ImportError:
        return ListDocumentsResponse(
            success=False,
            documents=[],
            total=0,
            limit=limit,
            offset=offset,
            error="Audit repository not available",
        )
    except Exception as e:
        logger.exception(f"Failed to list documents: {e}")
        return ListDocumentsResponse(
            success=False,
            documents=[],
            total=0,
            limit=limit,
            offset=offset,
            error=str(e),
        )


@router.get(
    "/documents/{file_hash}",
    response_model=GetDocumentResponse,
    responses=BASE_ERROR_RESPONSES,
    operation_id="getDocument",
    summary="Get document with history",
)
async def get_document(
    file_hash: str,
    org_id: str = Depends(get_org_id),
):
    """
    Get document details including processing jobs and content generations.

    Returns the document metadata along with its full processing history.

    **Multi-tenancy**: Scoped by X-Organization-ID header.
    """
    try:
        from src.db.repositories.audit import (
            get_document_by_name,
            get_jobs_by_document,
            get_generations_by_document,
        )

        # Get document - using file_hash as file_name for lookup
        doc = await get_document_by_name(file_hash)

        if not doc:
            return GetDocumentResponse(
                success=False,
                error=f"Document not found: {file_hash}"
            )

        document = DocumentRecord(
            organization_id=doc.get("organization_id"),
            file_hash=doc.get("file_hash", ""),
            storage_path=doc.get("storage_path", doc.get("file_path", "")),
            filename=doc.get("filename", doc.get("file_name", "")),
            file_size=doc.get("file_size", 0),
            created_at=doc.get("created_at"),
        )

        # Get associated jobs (org-scoped)
        doc_filename = doc.get("filename") or doc.get("file_name", "")
        jobs_data = await get_jobs_by_document(doc_filename, organization_id=org_id)
        jobs = [
            ProcessingJob(
                id=str(j["id"]),
                document_hash=j["document_hash"],
                file_name=j["file_name"],
                model=j["model"],
                complexity=j.get("complexity", "normal"),
                status=j["status"],
                started_at=j["started_at"],
                completed_at=j.get("completed_at"),
                cached=j.get("cached", False),
                output_path=j.get("output_path"),
                duration_ms=j.get("duration_ms"),
                error_message=j.get("error_message"),
            )
            for j in jobs_data
        ]

        # Get associated generations (org-scoped)
        gens_data = await get_generations_by_document(doc_filename, organization_id=org_id)
        generations = [
            DocumentGeneration(
                id=str(g["id"]),
                document_name=g["document_name"],
                document_hash=g.get("document_hash"),
                source_path=g.get("source_path"),
                generation_type=g["generation_type"],
                model=g["model"],
                processing_time_ms=g.get("processing_time_ms"),
                session_id=g.get("session_id"),
                created_at=g["created_at"],
            )
            for g in gens_data
        ]

        return GetDocumentResponse(
            success=True,
            document=document,
            jobs=jobs,
            generations=generations,
        )

    except ImportError:
        return GetDocumentResponse(
            success=False,
            error="Audit repository not available"
        )
    except Exception as e:
        logger.exception(f"Failed to get document: {e}")
        return GetDocumentResponse(
            success=False,
            error=str(e)
        )


@router.get(
    "/generations",
    response_model=ListGenerationsResponse,
    responses=BASE_ERROR_RESPONSES,
    operation_id="listGenerations",
    summary="List content generations",
)
async def list_generations(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    generation_type: Optional[str] = Query(default=None, description="Filter by type: summary, faqs, questions, all"),
    start_date: Optional[datetime] = Query(default=None, description="Filter generations after this date (ISO 8601)"),
    end_date: Optional[datetime] = Query(default=None, description="Filter generations before this date (ISO 8601)"),
    org_id: str = Depends(get_org_id),
):
    """
    List recent content generations (summaries, FAQs, questions) for the organization.

    Supports filtering by type and date range, with pagination.

    **Multi-tenancy**: Scoped by X-Organization-ID header.
    """
    try:
        from src.db.repositories.audit import get_recent_generations, count_generations

        gens = await get_recent_generations(
            limit=limit,
            offset=offset,
            organization_id=org_id,
            generation_type=generation_type,
            start_date=start_date,
            end_date=end_date,
        )

        total = await count_generations(
            organization_id=org_id,
            generation_type=generation_type,
            start_date=start_date,
            end_date=end_date,
        )

        generations = [
            DocumentGeneration(
                id=str(g["id"]),
                document_name=g["document_name"],
                document_hash=g.get("document_hash"),
                source_path=g.get("source_path"),
                generation_type=g["generation_type"],
                model=g["model"],
                processing_time_ms=g.get("processing_time_ms"),
                session_id=g.get("session_id"),
                created_at=g["created_at"],
            )
            for g in gens
        ]

        return ListGenerationsResponse(
            success=True,
            generations=generations,
            total=total,
            limit=limit,
            offset=offset,
        )

    except ImportError:
        return ListGenerationsResponse(
            success=False,
            generations=[],
            total=0,
            limit=limit,
            offset=offset,
            error="Audit repository not available"
        )
    except Exception as e:
        logger.exception(f"Failed to list generations: {e}")
        return ListGenerationsResponse(
            success=False,
            generations=[],
            total=0,
            limit=limit,
            offset=offset,
            error=str(e)
        )


@router.get(
    "/trail",
    response_model=AuditTrailResponse,
    responses=BASE_ERROR_RESPONSES,
    operation_id="getAuditTrail",
    summary="Get audit trail events",
)
async def get_audit_trail(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    event_type: Optional[str] = Query(default=None, description="Filter by event type"),
    file_name: Optional[str] = Query(default=None, description="Filter by file name"),
    start_date: Optional[datetime] = Query(default=None, description="Filter events after this date (ISO 8601)"),
    end_date: Optional[datetime] = Query(default=None, description="Filter events before this date (ISO 8601)"),
    org_id: str = Depends(get_org_id),
):
    """
    Get audit trail events for the organization.

    Supports filtering by event type, file name, and date range, with pagination.

    **Multi-tenancy**: Scoped by X-Organization-ID header.
    """
    try:
        from src.db.repositories.audit import get_audit_trail as get_trail, count_audit_events

        events_data = await get_trail(
            file_name=file_name,
            organization_id=org_id,
            event_type=event_type,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset,
        )

        total = await count_audit_events(
            organization_id=org_id,
            event_type=event_type,
            file_name=file_name,
            start_date=start_date,
            end_date=end_date,
        )

        events = [
            AuditEvent(
                id=str(e["id"]),
                created_at=e["created_at"],
                event_type=e.get("event_type") or "unknown",
                document_hash=e.get("document_hash"),
                file_name=e.get("file_name"),
                job_id=str(e["job_id"]) if e.get("job_id") else None,
                details=e.get("details") or {},
            )
            for e in events_data
        ]

        return AuditTrailResponse(
            success=True,
            events=events,
            total=total,
            limit=limit,
            offset=offset,
        )

    except ImportError:
        return AuditTrailResponse(
            success=False,
            events=[],
            total=0,
            limit=limit,
            offset=offset,
            error="Audit repository not available"
        )
    except Exception as e:
        logger.exception(f"Failed to get audit trail: {e}")
        return AuditTrailResponse(
            success=False,
            events=[],
            total=0,
            limit=limit,
            offset=offset,
            error=str(e)
        )


@router.post(
    "/events",
    response_model=LogEventResponse,
    responses=BASE_ERROR_RESPONSES,
    operation_id="logEvent",
    summary="Log custom audit event",
)
async def log_event(
    request: LogEventRequest,
    org_id: str = Depends(get_org_id),
):
    """
    Log a custom audit event for tracking and analytics.

    Useful for tracking application-specific events and user actions.

    **Multi-tenancy**: Event is associated with the requesting organization.
    """
    try:
        from src.db.repositories.audit import log_event as audit_log_event

        await audit_log_event(
            event_type=request.event_type,
            file_name=request.file_name,
            document_hash=request.document_hash,
            details=request.details,
            organization_id=org_id,
        )

        return LogEventResponse(
            success=True,
            event_id=None,  # log_event doesn't return an ID (fire-and-forget)
        )

    except ImportError:
        return LogEventResponse(
            success=False,
            error="Audit repository not available"
        )
    except Exception as e:
        logger.exception(f"Failed to log event: {e}")
        return LogEventResponse(
            success=False,
            error=str(e)
        )


# =============================================================================
# ACTIVITY TIMELINE ENDPOINT
# =============================================================================


@router.get(
    "/activity",
    response_model=ActivityTimelineResponse,
    responses=BASE_ERROR_RESPONSES,
    operation_id="getActivityTimeline",
    summary="Get activity timeline",
)
async def get_activity_timeline(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    event_type: Optional[str] = Query(default=None, description="Filter by event type"),
    file_name: Optional[str] = Query(default=None, description="Filter by file name"),
    start_date: Optional[datetime] = Query(default=None, description="Start date filter (ISO 8601)"),
    end_date: Optional[datetime] = Query(default=None, description="End date filter (ISO 8601)"),
    org_id: str = Depends(get_org_id),
):
    """
    Get activity timeline for the organization.

    Returns a chronological list of processing activities with
    human-readable timestamps and suggested UI elements (icons, colors).

    This endpoint is optimized for frontend dashboard display.

    **Multi-tenancy**: Scoped by X-Organization-ID header.
    """
    try:
        from src.db.repositories.audit import get_audit_trail as get_trail, count_audit_events

        events_data = await get_trail(
            file_name=file_name,
            organization_id=org_id,
            event_type=event_type,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset,
        )

        total = await count_audit_events(
            organization_id=org_id,
            event_type=event_type,
            file_name=file_name,
            start_date=start_date,
            end_date=end_date,
        )

        activities = [
            ActivityTimelineItem(
                id=str(e["id"]),
                timestamp=e["created_at"],
                timestamp_ago=format_time_ago(e["created_at"]) or "unknown",
                event_type=e.get("event_type") or "unknown",
                title=get_activity_title(e.get("event_type")),
                description=build_activity_description(
                    e.get("event_type"),
                    e.get("file_name"),
                    e.get("details"),
                ),
                file_name=e.get("file_name"),
                document_hash=e.get("document_hash"),
                status=e.get("details", {}).get("status"),
                status_color=get_status_color(e.get("details", {}).get("status")),
                icon=get_activity_icon(e.get("event_type")),
            )
            for e in events_data
        ]

        return ActivityTimelineResponse(
            success=True,
            activities=activities,
            total=total,
            limit=limit,
            offset=offset,
            has_more=(offset + len(activities)) < total,
            start_date=start_date,
            end_date=end_date,
        )

    except ImportError:
        return ActivityTimelineResponse(
            success=False,
            activities=[],
            total=0,
            limit=limit,
            offset=offset,
            error="Audit repository not available",
        )
    except Exception as e:
        logger.exception(f"Failed to get activity timeline: {e}")
        return ActivityTimelineResponse(
            success=False,
            activities=[],
            total=0,
            limit=limit,
            offset=offset,
            error=str(e),
        )
