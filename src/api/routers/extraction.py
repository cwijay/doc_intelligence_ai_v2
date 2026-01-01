"""Extraction API endpoints for document data extraction."""

import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status

from ..dependencies import get_org_id
from ..schemas.extraction import (
    AnalyzeFieldsRequest,
    AnalyzeFieldsResponse,
    GenerateSchemaRequest,
    GenerateSchemaResponse,
    ExtractDataRequest,
    ExtractDataResponse,
    TemplateListResponse,
    TemplateResponse,
    SaveExtractedDataRequest,
    SaveExtractedDataResponse,
    ExtractedRecordListResponse,
    ExtractedRecordDetailResponse,
    EXTRACTION_ERROR_RESPONSES,
)
from src.utils.timer_utils import elapsed_ms

router = APIRouter()
logger = logging.getLogger(__name__)


# =============================================================================
# GCS Cache for Extracted Data
# =============================================================================

def _build_extraction_cache_path(
    org_name: str,
    folder_name: str,
    document_name: str,
    template_name: str
) -> str:
    """Build GCS path for extraction cache.

    Path format: {org_name}/extracted/{folder_name}/{doc_base}_{template}.json

    Args:
        org_name: Organization name (e.g., "Acme corp")
        folder_name: Folder name (e.g., "invoices")
        document_name: Document filename (e.g., "IMG_4694.md")
        template_name: Template name (e.g., "Invoice Template")

    Returns:
        GCS path like "Acme corp/extracted/invoices/IMG_4694_invoice_template.json"
    """
    doc_base = Path(document_name).stem
    safe_template = template_name.strip().replace(' ', '_').lower()
    parts = [org_name, "extracted"]
    if folder_name:
        parts.append(folder_name)
    parts.append(f"{doc_base}_{safe_template}.json")
    return '/'.join(parts)


def _derive_folder_from_path(parsed_file_path: str) -> str:
    """Extract folder name from parsed file path.

    Example: "Acme corp/parsed/invoices/Sample1.md" -> "invoices"
    """
    if not parsed_file_path or 'parsed' not in parsed_file_path:
        return ""

    parts = parsed_file_path.split('/')
    try:
        parsed_idx = parts.index('parsed')
        folder_parts = parts[parsed_idx + 1:-1]  # Between 'parsed' and filename
        return '/'.join(folder_parts) if folder_parts else ""
    except ValueError:
        return ""


async def _check_extraction_cache(cache_path: str) -> Optional[Dict[str, Any]]:
    """Check if extraction exists in GCS cache.

    Args:
        cache_path: GCS path to cached extraction

    Returns:
        Cached extraction data if found, None otherwise
    """
    try:
        from src.storage.config import get_storage

        storage = get_storage()
        content = await storage.read(cache_path, use_prefix=False)

        if content:
            cached_data = json.loads(content)
            logger.info(f"Extraction cache HIT: {cache_path}")
            return cached_data

        return None

    except Exception as e:
        logger.debug(f"Extraction cache miss: {e}")
        return None


async def _save_extraction_cache(cache_path: str, extraction_data: Dict[str, Any]) -> bool:
    """Save extraction result to GCS cache.

    Args:
        cache_path: GCS path for cache
        extraction_data: Extracted data to cache

    Returns:
        True if saved successfully, False otherwise
    """
    try:
        from src.storage.config import get_storage

        storage = get_storage()
        content = json.dumps(extraction_data, indent=2, default=str, ensure_ascii=False)

        filename = Path(cache_path).name
        directory = str(Path(cache_path).parent)

        await storage.save(
            content=content,
            filename=filename,
            directory=directory,
            use_prefix=False
        )
        logger.info(f"Extraction cached: {cache_path}")
        return True

    except Exception as e:
        logger.warning(f"Failed to save extraction cache: {e}")
        return False

# Global agent instance (thread-safe singleton)
_extractor_agent = None
_agent_lock = None


async def get_extractor_agent():
    """Get or create ExtractorAgent instance (thread-safe)."""
    global _extractor_agent, _agent_lock

    import asyncio
    if _agent_lock is None:
        _agent_lock = asyncio.Lock()

    if _extractor_agent is None:
        async with _agent_lock:
            if _extractor_agent is None:
                from src.agents.extractor import ExtractorAgent, ExtractorAgentConfig
                config = ExtractorAgentConfig()
                _extractor_agent = ExtractorAgent(config)
                logger.info("ExtractorAgent initialized")

    return _extractor_agent


async def _load_document_content(parsed_file_path: str) -> str:
    """Load document content from GCS.

    Args:
        parsed_file_path: GCS path to parsed document

    Returns:
        Document content as string

    Raises:
        HTTPException: If document not found
    """
    try:
        from src.storage.config import get_storage

        storage = get_storage()
        content = await storage.read(parsed_file_path, use_prefix=False)

        if not content:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: {parsed_file_path}"
            )

        return content

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load document: {str(e)}"
        )


# =============================================================================
# Analyze Fields Endpoint
# =============================================================================

@router.post(
    "/analyze",
    response_model=AnalyzeFieldsResponse,
    responses=EXTRACTION_ERROR_RESPONSES,
    operation_id="analyzeDocumentFields",
    summary="Analyze document to discover extractable fields",
)
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
        content = await _load_document_content(request.parsed_file_path)

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
            # Convert agent response to API response
            return AnalyzeFieldsResponse(
                success=True,
                document_name=result.document_name,
                document_type=result.document_type,
                fields=[f.model_dump() for f in result.fields] if result.fields else None,
                has_line_items=result.has_line_items,
                line_item_fields=[f.model_dump() for f in result.line_item_fields] if result.line_item_fields else None,
                processing_time_ms=processing_time,
                session_id=result.session_id
            )
        else:
            return AnalyzeFieldsResponse(
                success=False,
                document_name=request.document_name,
                error=result.error,
                processing_time_ms=processing_time
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Field analysis failed: {e}")
        return AnalyzeFieldsResponse(
            success=False,
            document_name=request.document_name,
            error=str(e),
            processing_time_ms=elapsed_ms(start_time)
        )


# =============================================================================
# Generate Schema Endpoint
# =============================================================================

@router.post(
    "/generate-schema",
    response_model=GenerateSchemaResponse,
    responses=EXTRACTION_ERROR_RESPONSES,
    operation_id="generateExtractionSchema",
    summary="Generate and save extraction schema from selected fields",
)
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
                session_id=result.session_id
            )
        else:
            return GenerateSchemaResponse(
                success=False,
                template_name=request.template_name,
                error=result.error,
                processing_time_ms=processing_time
            )

    except Exception as e:
        logger.exception(f"Schema generation failed: {e}")
        return GenerateSchemaResponse(
            success=False,
            template_name=request.template_name,
            error=str(e),
            processing_time_ms=elapsed_ms(start_time)
        )


# =============================================================================
# List Templates Endpoint
# =============================================================================

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


# =============================================================================
# Get Template Endpoint
# =============================================================================

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


# =============================================================================
# Extract Data Endpoint
# =============================================================================

@router.post(
    "/extract",
    response_model=ExtractDataResponse,
    responses=EXTRACTION_ERROR_RESPONSES,
    operation_id="extractDocumentData",
    summary="Extract structured data from document using schema",
)
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
    from src.db.repositories.extraction_repository import get_organization_name

    start_time = time.time()

    try:
        # Resolve org_id to org_name for cache path
        org_name = await get_organization_name(org_id)
        if not org_name:
            org_name = org_id  # Fallback

        # Derive folder from parsed file path
        folder_name = _derive_folder_from_path(request.parsed_file_path)

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
        if template_name:
            cache_path = _build_extraction_cache_path(
                org_name=org_name,
                folder_name=folder_name,
                document_name=request.document_name,
                template_name=template_name
            )
            cached_data = await _check_extraction_cache(cache_path)

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
        content = await _load_document_content(request.parsed_file_path)

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
            if template_name:
                cache_data = {
                    "extraction_job_id": result.extraction_job_id,
                    "schema_title": result.schema_title,
                    "extracted_data": result.extracted_data,
                    "extracted_field_count": result.extracted_field_count,
                    "template_name": template_name,
                    "document_name": request.document_name,
                    "cached_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                }
                await _save_extraction_cache(cache_path, cache_data)

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


# =============================================================================
# Save Extracted Data Endpoint
# =============================================================================

@router.post(
    "/save",
    response_model=SaveExtractedDataResponse,
    responses=EXTRACTION_ERROR_RESPONSES,
    operation_id="saveExtractedData",
    summary="Save extracted data to database",
)
async def save_extracted_data(
    request: SaveExtractedDataRequest,
    agent=Depends(get_extractor_agent),
    org_id: str = Depends(get_org_id),
):
    """
    Save extracted data to the database.

    This persists the extraction results for later querying and analysis.
    Creates dynamic tables based on the template schema if they don't exist.

    **Multi-tenancy**: Data is saved with organization scoping.

    Returns:
        SaveExtractedDataResponse with record ID and table name
    """
    from src.db.repositories.extraction_repository import (
        save_extracted_record,
        get_organization_name,
        _get_table_name
    )

    try:
        # Use explicit folder_name or derive from source_file_path
        folder_name = request.folder_name
        if not folder_name and request.source_file_path:
            folder_name = _derive_folder_from_path(request.source_file_path)

        logger.info(f"[SAVE] Request: template={request.template_name}, folder={folder_name}, doc={request.document_id}")

        # Resolve org_id to org_name for table naming
        org_name = await get_organization_name(org_id)
        if not org_name:
            logger.warning(f"Organization name not found for {org_id}, using org_id")
            org_name = org_id  # Fallback to org_id if not found

        logger.info(f"[SAVE] Resolved org_name={org_name}")

        # Load template schema with folder_name
        schema = await agent.get_template(org_id, request.template_name, folder_name)
        logger.info(f"[SAVE] Template loaded: {bool(schema)}")

        if not schema:
            logger.error(f"[SAVE] Template not found: {request.template_name} in folder {folder_name}")
            return SaveExtractedDataResponse(
                success=False,
                message="Template not found",
                error=f"Template '{request.template_name}' not found in folder '{folder_name}'"
            )

        # Validate schema has properties
        if not schema.get("properties"):
            logger.error(f"[SAVE] Schema has no properties!")
            return SaveExtractedDataResponse(
                success=False,
                message="Invalid template",
                error="Template schema has no properties defined"
            )

        logger.info(f"[SAVE] Schema has {len(schema.get('properties', {}))} properties")

        # Save to database (creates tables if needed, uses UPSERT if entity ID found)
        save_result = await save_extracted_record(
            org_id=org_id,  # Still store org_id in records for querying
            org_name=org_name,  # Use org_name for table naming
            template_name=request.template_name,
            schema=schema,
            extraction_job_id=request.extraction_job_id,
            document_id=request.document_id,
            extracted_data=request.extracted_data,
            source_file_path=request.source_file_path
        )

        record_id = save_result["record_id"]
        was_updated = save_result["updated"]

        # Build table name for response
        table_name = _get_table_name(org_name, request.template_name)

        action = "Updated" if was_updated else "Saved"
        logger.info(f"{action} extracted data: job={request.extraction_job_id}, record={record_id}, table={table_name}")

        # Log audit event for successful save
        try:
            from src.agents.core.audit_queue import enqueue_audit_event

            enqueue_audit_event(
                event_type="extractor_data_saved",
                file_name=request.document_id,
                organization_id=org_id,
                details={
                    "record_id": record_id,
                    "table_name": table_name,
                    "template_name": request.template_name,
                    "extraction_job_id": request.extraction_job_id,
                    "action": "UPDATE" if was_updated else "CREATE",
                    "source_file_path": request.source_file_path
                }
            )
        except Exception as e:
            logger.warning(f"Failed to log audit event: {e}")

        return SaveExtractedDataResponse(
            success=True,
            record_id=record_id,
            table_name=table_name,
            message=f"Data {'updated' if was_updated else 'saved'} successfully"
        )

    except Exception as e:
        logger.exception(f"Failed to save extracted data: {e}")

        # Log failure audit event
        try:
            from src.agents.core.audit_queue import enqueue_audit_event

            enqueue_audit_event(
                event_type="extractor_save_failed",
                file_name=request.document_id,
                organization_id=org_id,
                details={
                    "template_name": request.template_name,
                    "extraction_job_id": request.extraction_job_id,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
        except Exception:
            pass

        return SaveExtractedDataResponse(
            success=False,
            message="Failed to save data",
            error=str(e)
        )


# =============================================================================
# List/Query Extracted Records Endpoints
# =============================================================================

@router.get(
    "/records/{template_name}",
    response_model=ExtractedRecordListResponse,
    responses=EXTRACTION_ERROR_RESPONSES,
    operation_id="listExtractedRecords",
    summary="List extracted records for a template",
)
async def list_extracted_records(
    template_name: str,
    limit: int = 100,
    offset: int = 0,
    org_id: str = Depends(get_org_id),
):
    """
    List all extracted records for a specific template.

    **Multi-tenancy**: Returns only records owned by the organization.

    Returns:
        ExtractedRecordListResponse with paginated records
    """
    from src.db.repositories.extraction_repository import (
        get_extracted_records,
        get_record_count,
        get_organization_name
    )

    try:
        # Resolve org_id to org_name for table naming
        org_name = await get_organization_name(org_id)
        if not org_name:
            org_name = org_id

        records = await get_extracted_records(
            org_id=org_id,
            template_name=template_name,
            limit=limit,
            offset=offset,
            org_name=org_name
        )

        total = await get_record_count(org_id, template_name, org_name=org_name)

        return ExtractedRecordListResponse(
            success=True,
            records=records,
            total=total,
            limit=limit,
            offset=offset
        )

    except Exception as e:
        logger.exception(f"Failed to list extracted records: {e}")
        return ExtractedRecordListResponse(
            success=False,
            error=str(e)
        )


@router.get(
    "/records/{template_name}/{record_id}",
    response_model=ExtractedRecordDetailResponse,
    responses=EXTRACTION_ERROR_RESPONSES,
    operation_id="getExtractedRecord",
    summary="Get extracted record with line items",
)
async def get_extracted_record(
    template_name: str,
    record_id: str,
    org_id: str = Depends(get_org_id),
):
    """
    Get a specific extracted record with its line items.

    **Multi-tenancy**: Only returns record if owned by the organization.

    Returns:
        ExtractedRecordDetailResponse with full record and line items
    """
    from src.db.repositories.extraction_repository import (
        get_extracted_record_with_line_items,
        get_organization_name
    )

    try:
        # Resolve org_id to org_name for table naming
        org_name = await get_organization_name(org_id)
        if not org_name:
            org_name = org_id

        result = await get_extracted_record_with_line_items(
            org_id=org_id,
            template_name=template_name,
            record_id=record_id,
            org_name=org_name
        )

        if not result:
            return ExtractedRecordDetailResponse(
                success=False,
                error=f"Record not found: {record_id}"
            )

        line_items = result.pop("line_items", [])

        return ExtractedRecordDetailResponse(
            success=True,
            record=result,
            line_items=line_items
        )

    except Exception as e:
        logger.exception(f"Failed to get extracted record: {e}")
        return ExtractedRecordDetailResponse(
            success=False,
            error=str(e)
        )


@router.delete(
    "/records/{template_name}/{record_id}",
    responses=EXTRACTION_ERROR_RESPONSES,
    operation_id="deleteExtractedRecord",
    summary="Delete extracted record",
)
async def delete_extracted_record_endpoint(
    template_name: str,
    record_id: str,
    org_id: str = Depends(get_org_id),
):
    """
    Delete an extracted record and its line items.

    **Multi-tenancy**: Only deletes if owned by the organization.

    Returns:
        Success status
    """
    from src.db.repositories.extraction_repository import (
        delete_extracted_record,
        get_organization_name
    )

    try:
        # Resolve org_id to org_name for table naming
        org_name = await get_organization_name(org_id)
        if not org_name:
            org_name = org_id

        deleted = await delete_extracted_record(
            org_id=org_id,
            template_name=template_name,
            record_id=record_id,
            org_name=org_name
        )

        if deleted:
            return {"success": True, "message": f"Record {record_id} deleted"}
        else:
            return {"success": False, "error": f"Record not found: {record_id}"}

    except Exception as e:
        logger.exception(f"Failed to delete extracted record: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# Export to Excel Endpoint
# =============================================================================

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
    from fastapi.responses import StreamingResponse
    import io

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


# =============================================================================
# Health Check
# =============================================================================

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
