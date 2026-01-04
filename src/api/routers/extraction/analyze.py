"""Analyze fields endpoint for extraction router."""

import logging
import time

from fastapi import APIRouter, Depends

from src.api.dependencies import get_org_id, get_extractor_agent
from src.core.usage import check_quota, track_tokens
from src.api.schemas.extraction import (
    AnalyzeFieldsRequest,
    AnalyzeFieldsResponse,
    EXTRACTION_ERROR_RESPONSES,
)
from src.utils.timer_utils import elapsed_ms

from .helpers import load_document_content, map_token_usage

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/analyze",
    response_model=AnalyzeFieldsResponse,
    responses=EXTRACTION_ERROR_RESPONSES,
    operation_id="analyzeDocumentFields",
    summary="Analyze document to discover extractable fields",
)
@check_quota(usage_type="tokens", estimated_usage=2000)
@track_tokens(feature="extractor_agent", tokens_attr="token_usage.total_tokens")
async def analyze_document_fields(
    request: AnalyzeFieldsRequest,
    agent=Depends(get_extractor_agent),
    org_id: str = Depends(get_org_id),
):
    """
    Analyze a parsed document to discover all extractable fields.

    This is typically the first step in the extraction workflow:
    1. Analyze document to discover fields
    2. User selects which fields to extract
    3. Generate schema from selected fields
    4. Extract data using schema

    **Multi-tenancy**: Scoped by X-Organization-ID header.

    Returns:
        AnalyzeFieldsResponse with discovered fields and their properties
    """
    start_time = time.time()

    try:
        # Load document content from GCS
        content = await load_document_content(request.parsed_file_path)

        # Analyze fields
        result = await agent.analyze_fields(
            content=content,
            document_name=request.document_name,
            document_type_hint=request.document_type_hint,
            organization_id=org_id,
            session_id=request.session_id
        )

        processing_time = elapsed_ms(start_time)

        if result.success:
            return AnalyzeFieldsResponse(
                success=True,
                document_name=result.document_name,
                document_type=result.document_type,
                fields=[f.model_dump() for f in result.fields] if result.fields else None,
                has_line_items=result.has_line_items,
                line_item_fields=[f.model_dump() for f in result.line_item_fields] if result.line_item_fields else None,
                processing_time_ms=processing_time,
                session_id=result.session_id,
                token_usage=map_token_usage(result.token_usage)
            )
        else:
            return AnalyzeFieldsResponse(
                success=False,
                document_name=request.document_name,
                error=result.error,
                processing_time_ms=processing_time
            )

    except Exception as e:
        logger.exception(f"Field analysis failed: {e}")
        return AnalyzeFieldsResponse(
            success=False,
            document_name=request.document_name,
            error=str(e),
            processing_time_ms=elapsed_ms(start_time)
        )
