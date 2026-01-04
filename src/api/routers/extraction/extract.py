"""Extract data endpoint for extraction router."""

import logging
import time

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_org_id, get_extractor_agent
from src.core.usage import check_quota, track_tokens
from src.api.schemas.extraction import (
    ExtractDataRequest,
    ExtractDataResponse,
    EXTRACTION_ERROR_RESPONSES,
)
from src.utils.timer_utils import elapsed_ms

from .cache import (
    build_extraction_cache_path,
    derive_folder_from_path,
    check_extraction_cache,
    save_extraction_cache,
)
from .helpers import load_document_content, get_organization_name_safe

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/extract",
    response_model=ExtractDataResponse,
    responses=EXTRACTION_ERROR_RESPONSES,
    operation_id="extractDocumentData",
    summary="Extract structured data from document using schema",
)
@check_quota(usage_type="tokens", estimated_usage=3000)
@track_tokens(feature="extractor_agent", tokens_attr="token_usage.total_tokens")
async def extract_document_data(
    request: ExtractDataRequest,
    agent=Depends(get_extractor_agent),
    org_id: str = Depends(get_org_id),
):
    """
    Extract structured data from a document using a schema.

    Either provide a `template_name` to use a saved template, or provide
    a `schema` directly.

    **Caching**: If document was previously extracted with the same template,
    cached result is returned (cached=true).

    **Multi-tenancy**: Scoped by X-Organization-ID header.

    Returns:
        ExtractDataResponse with extracted data
    """
    start_time = time.time()

    try:
        # Resolve org_id to org_name for cache path
        org_name = await get_organization_name_safe(org_id)

        # Derive folder from parsed file path
        folder_name = derive_folder_from_path(request.parsed_file_path)

        # Get template name for cache key
        template_name = request.template_name
        schema = request.schema_definition

        if not schema and template_name:
            # Load template with folder_name for correct GCS path
            schema = await agent.get_template(org_id, template_name, folder_name)
            if not schema:
                logger.warning(f"Template not found: {template_name} in folder {folder_name}")
                return ExtractDataResponse(
                    success=False,
                    document_name=request.document_name,
                    error=f"Template not found: {request.template_name} in folder '{folder_name}'",
                    processing_time_ms=elapsed_ms(start_time)
                )
        elif schema and not template_name:
            # Extract template name from schema if provided directly
            template_name = schema.get("title", "custom_schema")

        if not schema:
            return ExtractDataResponse(
                success=False,
                document_name=request.document_name,
                error="Either template_name or schema must be provided",
                processing_time_ms=elapsed_ms(start_time)
            )

        # Check GCS cache first
        cache_path = None
        if template_name:
            cache_path = build_extraction_cache_path(
                org_name=org_name,
                folder_name=folder_name,
                document_name=request.document_name,
                template_name=template_name
            )
            cached_data = await check_extraction_cache(cache_path)

            if cached_data:
                processing_time = elapsed_ms(start_time)
                return ExtractDataResponse(
                    success=True,
                    extraction_job_id=cached_data.get("extraction_job_id"),
                    document_name=request.document_name,
                    schema_title=cached_data.get("schema_title"),
                    extracted_data=cached_data.get("extracted_data"),
                    extracted_field_count=cached_data.get("extracted_field_count", 0),
                    token_usage=None,  # No token usage for cached results
                    cached=True,
                    processing_time_ms=processing_time,
                    session_id=request.session_id
                )

        # Load document content and extract
        content = await load_document_content(request.parsed_file_path)

        result = await agent.extract_data(
            content=content,
            schema=schema,
            document_name=request.document_name,
            organization_id=org_id,
            session_id=request.session_id
        )

        processing_time = elapsed_ms(start_time)

        if result.success:
            # Save to cache for future requests
            if template_name and cache_path:
                cache_data = {
                    "extraction_job_id": result.extraction_job_id,
                    "schema_title": result.schema_title,
                    "extracted_data": result.extracted_data,
                    "extracted_field_count": result.extracted_field_count,
                    "template_name": template_name,
                    "document_name": request.document_name,
                    "cached_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                }
                await save_extraction_cache(cache_path, cache_data)

            return ExtractDataResponse(
                success=True,
                extraction_job_id=result.extraction_job_id,
                document_name=result.document_name,
                schema_title=result.schema_title,
                extracted_data=result.extracted_data,
                extracted_field_count=result.extracted_field_count,
                token_usage=result.token_usage.model_dump() if result.token_usage else None,
                cached=False,
                processing_time_ms=processing_time,
                session_id=result.session_id
            )
        else:
            return ExtractDataResponse(
                success=False,
                document_name=request.document_name,
                error=result.error,
                processing_time_ms=processing_time
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Data extraction failed: {e}")
        return ExtractDataResponse(
            success=False,
            document_name=request.document_name,
            error=str(e),
            processing_time_ms=elapsed_ms(start_time)
        )
