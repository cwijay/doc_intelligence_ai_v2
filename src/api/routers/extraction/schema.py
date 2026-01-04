"""Schema and template endpoints for extraction router."""

import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends

from src.api.dependencies import get_org_id, get_extractor_agent
from src.core.usage import check_quota, track_tokens
from src.api.schemas.extraction import (
    GenerateSchemaRequest,
    GenerateSchemaResponse,
    TemplateListResponse,
    TemplateResponse,
    EXTRACTION_ERROR_RESPONSES,
)
from src.utils.timer_utils import elapsed_ms

from .helpers import map_token_usage

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/generate-schema",
    response_model=GenerateSchemaResponse,
    responses=EXTRACTION_ERROR_RESPONSES,
    operation_id="generateExtractionSchema",
    summary="Generate and save extraction schema from selected fields",
)
@check_quota(usage_type="tokens", estimated_usage=1500)
@track_tokens(feature="extractor_agent", tokens_attr="token_usage.total_tokens")
async def generate_extraction_schema(
    request: GenerateSchemaRequest,
    agent=Depends(get_extractor_agent),
    org_id: str = Depends(get_org_id),
):
    """
    Generate a JSON schema from user-selected fields.

    The schema can be saved as a reusable template for future extractions
    of similar documents.

    **Multi-tenancy**: Schema is saved in organization-specific GCS path.

    Returns:
        GenerateSchemaResponse with generated schema and GCS URI
    """
    start_time = time.time()

    try:
        # Convert to list of dicts for agent
        selected_fields = [f.model_dump() for f in request.selected_fields]

        result = await agent.generate_schema(
            selected_fields=selected_fields,
            template_name=request.template_name,
            document_type=request.document_type,
            organization_id=org_id,
            folder_name=request.folder_name,
            save_to_gcs=request.save_template,
            session_id=request.session_id
        )

        processing_time = elapsed_ms(start_time)

        if result.success:
            return GenerateSchemaResponse(
                success=True,
                template_name=result.template_name,
                document_type=result.document_type,
                schema_definition=result.schema_definition,
                gcs_uri=result.gcs_uri,
                processing_time_ms=processing_time,
                session_id=result.session_id,
                token_usage=map_token_usage(result.token_usage)
            )
        else:
            return GenerateSchemaResponse(
                success=False,
                template_name=request.template_name,
                error=result.error,
                processing_time_ms=processing_time,
                token_usage=map_token_usage(result.token_usage)
            )

    except Exception as e:
        logger.exception(f"Schema generation failed: {e}")
        return GenerateSchemaResponse(
            success=False,
            template_name=request.template_name,
            error=str(e),
            processing_time_ms=elapsed_ms(start_time)
        )


@router.get(
    "/templates",
    response_model=TemplateListResponse,
    responses=EXTRACTION_ERROR_RESPONSES,
    operation_id="listExtractionTemplates",
    summary="List all extraction templates for organization",
)
async def list_extraction_templates(
    agent=Depends(get_extractor_agent),
    org_id: str = Depends(get_org_id),
):
    """
    List all saved extraction templates for the organization.

    Templates are reusable schemas that can be applied to similar documents.

    **Multi-tenancy**: Returns only templates owned by the organization.

    Returns:
        TemplateListResponse with list of template metadata
    """
    try:
        templates = await agent.list_templates(org_id)

        return TemplateListResponse(
            success=True,
            templates=templates,
            total=len(templates)
        )

    except Exception as e:
        logger.exception(f"Failed to list templates: {e}")
        return TemplateListResponse(
            success=False,
            error=str(e),
            total=0
        )


@router.get(
    "/templates/{template_name}",
    response_model=TemplateResponse,
    responses=EXTRACTION_ERROR_RESPONSES,
    operation_id="getExtractionTemplate",
    summary="Get extraction template by name",
)
async def get_extraction_template(
    template_name: str,
    folder_name: Optional[str] = None,
    agent=Depends(get_extractor_agent),
    org_id: str = Depends(get_org_id),
):
    """
    Get a specific extraction template by name.

    **Multi-tenancy**: Only returns template if owned by the organization.

    Args:
        template_name: Name of the template
        folder_name: Optional folder where template is stored

    Returns:
        TemplateResponse with full schema
    """
    try:
        template = await agent.get_template(org_id, template_name, folder_name)

        if template:
            return TemplateResponse(
                success=True,
                name=template.get("title", template_name),
                document_type=template.get("metadata", {}).get("document_type"),
                schema_definition=template
            )
        else:
            return TemplateResponse(
                success=False,
                error=f"Template not found: {template_name} in folder '{folder_name}'"
            )

    except Exception as e:
        logger.exception(f"Failed to get template: {e}")
        return TemplateResponse(
            success=False,
            error=str(e)
        )
