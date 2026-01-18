"""Export endpoints for extraction router."""

import io
import json
import logging
from typing import Optional, Dict, Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from src.api.dependencies import get_org_id
from src.api.schemas.extraction import EXTRACTION_ERROR_RESPONSES
from src.storage.config import get_storage

from .helpers import get_organization_name_safe

router = APIRouter()
logger = logging.getLogger(__name__)

# Common business ID field names to look for in extracted data
BUSINESS_ID_FIELDS = [
    "invoice_id", "invoice_number", "invoice_no",
    "order_id", "order_number", "order_no",
    "po_number", "po_id", "purchase_order_number",
    "contract_id", "contract_number", "contract_no",
    "receipt_id", "receipt_number", "receipt_no",
    "quote_id", "quote_number",
    "bill_id", "bill_number",
    "reference_number", "reference_id", "ref_number",
    "document_id", "doc_id", "doc_number",
    "transaction_id", "transaction_number",
    "claim_id", "claim_number",
    "policy_number", "policy_id",
    "account_number", "account_id",
    "customer_id", "vendor_id",
]


def _find_business_id(extracted_data: Dict[str, Any]) -> Optional[str]:
    """
    Find a business identifier from the extracted data.

    Searches for common ID fields like invoice_number, order_id, etc.

    Args:
        extracted_data: The extracted data dictionary

    Returns:
        The business ID value if found, None otherwise
    """
    # Create lowercase key mapping for case-insensitive matching
    lowercase_keys = {k.lower(): k for k in extracted_data.keys()}

    for field_name in BUSINESS_ID_FIELDS:
        # Check for exact match (case-insensitive)
        if field_name in lowercase_keys:
            original_key = lowercase_keys[field_name]
            value = extracted_data.get(original_key)
            if value is not None and str(value).strip():
                return str(value).strip()

    return None


def _sanitize_filename(name: str) -> str:
    """
    Sanitize a string for use in a filename.

    Removes or replaces characters that are invalid in filenames.
    """
    import re
    # Replace spaces and common separators with underscores
    name = re.sub(r'[\s\-]+', '_', name)
    # Remove characters that are invalid in filenames
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', name)
    # Remove leading/trailing underscores and dots
    name = name.strip('_.')
    # Limit length
    return name[:100] if name else "export"


async def _find_extraction_by_job_id(
    org_name: str,
    extraction_job_id: str
) -> Optional[Dict[str, Any]]:
    """
    Search org's GCS extraction cache for file with matching extraction_job_id.

    Args:
        org_name: Organization name
        extraction_job_id: The extraction job ID to find

    Returns:
        Cached extraction data if found, None otherwise
    """
    storage = get_storage()
    search_prefix = f"{org_name}/extracted/"

    try:
        # List all files in the org's extracted folder (returns gs:// URIs)
        files = await storage.list_files(search_prefix, extension=".json", use_prefix=False)

        for file_uri in files:
            try:
                # Read file content - storage.read handles gs:// URIs
                content = await storage.read(file_uri, use_prefix=False)
                if not content:
                    continue

                data = json.loads(content)
                if data.get("extraction_job_id") == extraction_job_id:
                    logger.info(f"Found extraction job {extraction_job_id} at {file_uri}")
                    return data

            except (json.JSONDecodeError, Exception) as e:
                logger.debug(f"Failed to read extraction file {file_uri}: {e}")
                continue

        return None

    except Exception as e:
        logger.error(f"Failed to search extractions for org {org_name}: {e}")
        return None


def _create_excel_from_extraction(
    extraction_data: Dict[str, Any],
    document_name: str
) -> io.BytesIO:
    """
    Create Excel file from extracted data.

    Creates multiple sheets:
    - Header: Main extracted fields
    - Line Items: If line_items array exists

    Args:
        extraction_data: The extracted_data dict from extraction
        document_name: Name of the source document

    Returns:
        BytesIO buffer containing the Excel file
    """
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Extract header fields (non-line-items)
        header_data = {}
        line_items = None

        for key, value in extraction_data.items():
            if key == "line_items" and isinstance(value, list):
                line_items = value
            elif not isinstance(value, (list, dict)):
                header_data[key] = value
            elif isinstance(value, dict):
                # Flatten nested dicts with prefix
                for nested_key, nested_value in value.items():
                    if not isinstance(nested_value, (list, dict)):
                        header_data[f"{key}_{nested_key}"] = nested_value

        # Create header sheet
        if header_data:
            # Convert to DataFrame with single row, transpose for better readability
            header_df = pd.DataFrame([header_data])
            header_df.to_excel(writer, sheet_name='Extracted Data', index=False)

            # Also create a vertical view for better readability
            vertical_df = pd.DataFrame({
                'Field': list(header_data.keys()),
                'Value': list(header_data.values())
            })
            vertical_df.to_excel(writer, sheet_name='Fields', index=False)

        # Create line items sheet if present
        if line_items and len(line_items) > 0:
            # Flatten each line item
            flattened_items = []
            for item in line_items:
                flat_item = {}
                for key, value in item.items():
                    if not isinstance(value, (list, dict)):
                        flat_item[key] = value
                flattened_items.append(flat_item)

            line_items_df = pd.DataFrame(flattened_items)
            line_items_df.to_excel(writer, sheet_name='Line Items', index=False)

        # Add metadata sheet
        metadata = {
            'Field': ['Document Name', 'Export Date'],
            'Value': [document_name, pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')]
        }
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_excel(writer, sheet_name='Metadata', index=False)

    output.seek(0)
    return output


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

    Searches the organization's extraction cache for the specified job ID
    and exports the extracted data to an Excel file.

    **Multi-tenancy**: Only exports data owned by the organization.

    Returns:
        Excel file as streaming response
    """
    try:
        # Resolve org_id to org_name for GCS path
        org_name = await get_organization_name_safe(org_id)

        # Find the extraction data by job ID
        cached_data = await _find_extraction_by_job_id(org_name, extraction_job_id)

        if not cached_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Extraction job not found: {extraction_job_id}"
            )

        extracted_data = cached_data.get("extracted_data")
        if not extracted_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No extracted data found for this job"
            )

        document_name = cached_data.get("document_name", "extraction")
        template_name = cached_data.get("template_name", "export")

        # Create Excel file
        excel_buffer = _create_excel_from_extraction(extracted_data, document_name)

        # Generate filename: {template_name}_{business_id}.xlsx
        # Find business ID from extracted data (invoice_number, order_id, etc.)
        business_id = _find_business_id(extracted_data)

        # Sanitize template name for filename
        safe_template = _sanitize_filename(template_name)

        if business_id:
            safe_business_id = _sanitize_filename(business_id)
            filename = f"{safe_template}_{safe_business_id}.xlsx"
        else:
            # Fallback to template_name_document_name if no business ID found
            safe_doc_name = _sanitize_filename(document_name)
            if safe_doc_name.endswith('.md'):
                safe_doc_name = safe_doc_name[:-3]
            filename = f"{safe_template}_{safe_doc_name}.xlsx"

        logger.info(f"Exporting extraction {extraction_job_id} as {filename}")

        return StreamingResponse(
            excel_buffer,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Excel export failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
