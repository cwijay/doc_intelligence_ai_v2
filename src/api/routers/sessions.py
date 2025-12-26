"""Session Management API endpoints.

Multi-tenancy: All endpoints are scoped by organization_id from request headers.
Sessions must belong to the requesting organization.
"""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Path

from ..dependencies import get_session_manager, get_document_agent, get_sheets_agent, get_org_id
from ..schemas.common import SessionInfo, SuccessResponse
from ..schemas.errors import BASE_ERROR_RESPONSES

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/{session_id}",
    response_model=SessionInfo,
    responses=BASE_ERROR_RESPONSES,
    operation_id="getSession",
    summary="Get session information",
)
async def get_session(
    session_id: str = Path(..., description="Session ID"),
    org_id: str = Depends(get_org_id),
):
    """
    Get information about an active session.

    Returns session statistics including query count, token usage, and processing time.

    **Session timeout**: Sessions expire after 30 minutes of inactivity.

    **Multi-tenancy**: Validates session belongs to the requesting organization.
    """
    sessions = get_session_manager()

    if session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found"
        )

    info = sessions[session_id]

    # Multi-tenancy: Validate session belongs to requesting org
    session_org_id = info.get("organization_id")
    if session_org_id and session_org_id != org_id:
        raise HTTPException(
            status_code=403,
            detail=f"Session '{session_id}' belongs to another organization"
        )

    return SessionInfo(
        session_id=session_id,
        user_id=info.get("user_id"),
        created_at=info.get("created_at", datetime.utcnow()),
        last_activity=info.get("last_activity", datetime.utcnow()),
        expires_at=info.get("expires_at", datetime.utcnow()),
        query_count=info.get("query_count", 0),
        total_tokens_used=info.get("total_tokens_used", 0),
        total_processing_time_ms=info.get("total_processing_time_ms", 0),
    )


@router.delete(
    "/{session_id}",
    response_model=SuccessResponse,
    responses=BASE_ERROR_RESPONSES,
    operation_id="deleteSession",
    summary="End and cleanup session",
)
async def end_session(
    session_id: str = Path(..., description="Session ID"),
    save_summary: bool = True,
    org_id: str = Depends(get_org_id),
):
    """
    End a session and clean up associated resources.

    **Summary**: Optionally save a conversation summary before ending.

    **Cleanup**: Releases memory, caches, and agent resources for the session.

    **Multi-tenancy**: Validates session belongs to the requesting organization.
    """
    try:
        sessions = get_session_manager()

        # Multi-tenancy: Validate session belongs to requesting org before deletion
        if session_id in sessions:
            session_org_id = sessions[session_id].get("organization_id")
            if session_org_id and session_org_id != org_id:
                raise HTTPException(
                    status_code=403,
                    detail=f"Session '{session_id}' belongs to another organization"
                )

        # Try to end session in document agent
        try:
            doc_agent = await get_document_agent()
            await doc_agent.end_session(
                session_id=session_id,
                save_summary=save_summary,
            )
            logger.info(f"Ended document agent session: {session_id}")
        except Exception as e:
            logger.debug(f"Document agent session cleanup: {e}")

        # Try to end session in sheets agent
        try:
            sheets_agent = await get_sheets_agent()
            await sheets_agent.end_session(
                session_id=session_id,
                save_summary=save_summary,
            )
            logger.info(f"Ended sheets agent session: {session_id}")
        except Exception as e:
            logger.debug(f"Sheets agent session cleanup: {e}")

        # Remove from session manager
        if session_id in sessions:
            del sessions[session_id]

        return SuccessResponse(
            success=True,
            message=f"Session '{session_id}' ended successfully"
        )

    except HTTPException:
        raise  # Re-raise HTTP exceptions (like 403)
    except Exception as e:
        logger.exception(f"Failed to end session: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
