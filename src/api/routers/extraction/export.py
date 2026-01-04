"""Export endpoints for extraction router."""

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import get_org_id
from src.api.schemas.extraction import EXTRACTION_ERROR_RESPONSES

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get(
    "/export/{extraction_job_id}",
    responses={
        200: {"content": {"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": {}}},
        **EXTRACTION_ERROR_RESPONSES
    },
    operation_id="exportExtractedDataToExcel",
    summary="Export extracted data as Excel file",
)
async def export_to_excel(
    extraction_job_id: str,
    org_id: str = Depends(get_org_id),
):
    """
    Export extracted data as an Excel file.

    **Multi-tenancy**: Only exports data owned by the organization.

    Returns:
        Excel file as streaming response
    """
    try:
        # TODO: Implement Excel export
        # For now, return a placeholder error
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Excel export not yet implemented"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Excel export failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
