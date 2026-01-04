"""Health check endpoint for extraction router."""

import logging

from fastapi import APIRouter, Depends

from src.api.dependencies import get_extractor_agent

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get(
    "/health",
    operation_id="extractorHealth",
    summary="Check extractor agent health",
)
async def extractor_health(
    agent=Depends(get_extractor_agent),
):
    """Get health status of the extractor agent."""
    try:
        return agent.get_health_status()
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
