"""Audit and Analytics API endpoints.

Multi-tenancy: All endpoints are scoped by organization_id from request headers.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query

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
)

logger = logging.getLogger(__name__)
router = APIRouter()


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
    status: Optional[str] = Query(default=None, description="Filter by status: pending, running, completed, failed"),
    org_id: str = Depends(get_org_id),
):
    """
    List recent document processing jobs for the organization.

    Includes job status, duration, model used, and any error messages.

    **Multi-tenancy**: Scoped by X-Organization-ID header.
    """
    try:
        from src.db.repositories.audit_repository import get_processing_history

        jobs = await get_processing_history(limit=limit, organization_id=org_id)

        # Filter by status if provided
        if status:
            jobs = [j for j in jobs if j.get("status") == status]

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
            total=len(result_jobs),
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
async def get_job(job_id: str):
    """
    Get details of a specific processing job.

    **Note**: This endpoint is not yet implemented.
    """
    raise HTTPException(
        status_code=501,
        detail="Get job by ID is not yet implemented. Use /jobs to list all jobs."
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
):
    """
    List all registered documents with their metadata.

    **Note**: This endpoint is not yet implemented.
    """
    raise HTTPException(
        status_code=501,
        detail="List documents is not yet implemented. Document metadata is available via /trail endpoint."
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
        from src.db.repositories.audit_repository import (
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
            file_hash=doc["file_hash"],
            file_path=doc["file_path"],
            file_name=doc["file_name"],
            file_size=doc["file_size"],
            created_at=doc["created_at"],
        )

        # Get associated jobs (org-scoped)
        jobs_data = await get_jobs_by_document(doc["file_name"], organization_id=org_id)
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
        gens_data = await get_generations_by_document(doc["file_name"], organization_id=org_id)
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
    generation_type: Optional[str] = Query(default=None, description="Filter by type: summary, faqs, questions"),
    org_id: str = Depends(get_org_id),
):
    """
    List recent content generations (summaries, FAQs, questions) for the organization.

    Includes generation type, model used, and processing time.

    **Multi-tenancy**: Scoped by X-Organization-ID header.
    """
    try:
        from src.db.repositories.audit_repository import get_recent_generations

        gens = await get_recent_generations(limit=limit, organization_id=org_id)

        # Filter by type if provided
        if generation_type:
            gens = [g for g in gens if g.get("generation_type") == generation_type]

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
            total=len(generations),
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
    org_id: str = Depends(get_org_id),
):
    """
    Get audit trail events for the organization.

    Provides a chronological log of all document processing activities.

    **Multi-tenancy**: Scoped by X-Organization-ID header.
    """
    try:
        from src.db.repositories.audit_repository import get_audit_trail as get_trail

        events_data = await get_trail(file_name=file_name, organization_id=org_id)

        # Filter by event type if provided
        if event_type:
            events_data = [e for e in events_data if e.get("event_type") == event_type]

        # Apply limit after filtering
        events_data = events_data[:limit]

        events = [
            AuditEvent(
                id=str(e["id"]),
                timestamp=e["timestamp"],
                event_type=e["event_type"],
                document_hash=e.get("document_hash"),
                file_name=e.get("file_name"),
                job_id=str(e["job_id"]) if e.get("job_id") else None,
                details=e.get("details", {}),
            )
            for e in events_data
        ]

        return AuditTrailResponse(
            success=True,
            events=events,
            total=len(events),
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
        from src.db.repositories.audit_repository import log_event as audit_log_event

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
