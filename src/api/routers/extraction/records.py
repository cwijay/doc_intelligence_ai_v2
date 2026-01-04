"""Records management endpoints for extraction router."""

import logging

from fastapi import APIRouter, Depends

from src.api.dependencies import get_org_id, get_extractor_agent
from src.api.schemas.extraction import (
    SaveExtractedDataRequest,
    SaveExtractedDataResponse,
    ExtractedRecordListResponse,
    ExtractedRecordDetailResponse,
    EXTRACTION_ERROR_RESPONSES,
)

from .cache import derive_folder_from_path
from .helpers import get_organization_name_safe

router = APIRouter()
logger = logging.getLogger(__name__)


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
        _get_table_name
    )

    try:
        # Use explicit folder_name or derive from source_file_path
        folder_name = request.folder_name
        if not folder_name and request.source_file_path:
            folder_name = derive_folder_from_path(request.source_file_path)

        logger.info(f"[SAVE] Request: template={request.template_name}, folder={folder_name}, doc={request.document_id}")

        # Resolve org_id to org_name for table naming
        org_name = await get_organization_name_safe(org_id)
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
        _log_save_audit_event(
            org_id=org_id,
            document_id=request.document_id,
            record_id=record_id,
            table_name=table_name,
            template_name=request.template_name,
            extraction_job_id=request.extraction_job_id,
            source_file_path=request.source_file_path,
            was_updated=was_updated
        )

        return SaveExtractedDataResponse(
            success=True,
            record_id=record_id,
            table_name=table_name,
            message=f"Data {'updated' if was_updated else 'saved'} successfully"
        )

    except Exception as e:
        logger.exception(f"Failed to save extracted data: {e}")

        # Log failure audit event
        _log_save_failure_audit_event(
            org_id=org_id,
            document_id=request.document_id,
            template_name=request.template_name,
            extraction_job_id=request.extraction_job_id,
            error=e
        )

        return SaveExtractedDataResponse(
            success=False,
            message="Failed to save data",
            error=str(e)
        )


def _log_save_audit_event(
    org_id: str,
    document_id: str,
    record_id: str,
    table_name: str,
    template_name: str,
    extraction_job_id: str,
    source_file_path: str,
    was_updated: bool
):
    """Log audit event for successful save."""
    try:
        from src.agents.core.audit_queue import enqueue_audit_event

        enqueue_audit_event(
            event_type="extractor_data_saved",
            file_name=document_id,
            organization_id=org_id,
            details={
                "record_id": record_id,
                "table_name": table_name,
                "template_name": template_name,
                "extraction_job_id": extraction_job_id,
                "action": "UPDATE" if was_updated else "CREATE",
                "source_file_path": source_file_path
            }
        )
    except Exception as e:
        logger.warning(f"Failed to log audit event: {e}")


def _log_save_failure_audit_event(
    org_id: str,
    document_id: str,
    template_name: str,
    extraction_job_id: str,
    error: Exception
):
    """Log audit event for save failure."""
    try:
        from src.agents.core.audit_queue import enqueue_audit_event

        enqueue_audit_event(
            event_type="extractor_save_failed",
            file_name=document_id,
            organization_id=org_id,
            details={
                "template_name": template_name,
                "extraction_job_id": extraction_job_id,
                "error": str(error),
                "error_type": type(error).__name__
            }
        )
    except Exception:
        pass


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
    )

    try:
        # Resolve org_id to org_name for table naming
        org_name = await get_organization_name_safe(org_id)

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
    from src.db.repositories.extraction_repository import get_extracted_record_with_line_items

    try:
        # Resolve org_id to org_name for table naming
        org_name = await get_organization_name_safe(org_id)

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
    from src.db.repositories.extraction_repository import delete_extracted_record

    try:
        # Resolve org_id to org_name for table naming
        org_name = await get_organization_name_safe(org_id)

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
