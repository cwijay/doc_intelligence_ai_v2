"""Content API endpoints for loading pre-parsed document content."""

import logging
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ..dependencies import get_org_id
from ..schemas.content import (
    LoadParsedRequest,
    LoadParsedResponse,
    CheckParsedExistsResponse,
    CONTENT_ERROR_RESPONSES,
)
from src.storage.config import get_storage

router = APIRouter()
logger = logging.getLogger(__name__)


def _construct_parsed_path(org_name: str, folder_name: str, document_name: str) -> str:
    """Construct the GCS path for parsed document.

    Format: {org_name}/parsed/{folder_name}/{document_name}.md
    """
    # Strip original extension and add .md
    doc_path = Path(document_name)
    markdown_name = f"{doc_path.stem}.md"

    return f"{org_name}/parsed/{folder_name}/{markdown_name}"


def _construct_original_path(org_name: str, folder_name: str, document_name: str) -> str:
    """Construct the GCS path for original document.

    Format: {org_name}/original/{folder_name}/{document_name}
    """
    return f"{org_name}/original/{folder_name}/{document_name}"


@router.post(
    "/load-parsed",
    response_model=LoadParsedResponse,
    responses=CONTENT_ERROR_RESPONSES,
    operation_id="loadParsedContent",
    summary="Load pre-parsed document content from GCS",
)
async def load_parsed_content(
    request: LoadParsedRequest,
    org_id: str = Depends(get_org_id),
) -> LoadParsedResponse:
    """
    Load previously parsed document content from GCS.

    This endpoint retrieves parsed markdown content that was created
    by a previous parsing operation. Returns 404 if the parsed file
    does not exist in GCS.

    **Use case**: When document parsing times out but the parsed content
    already exists in GCS from a previous attempt.

    **Multi-tenancy**: Scoped by X-Organization-ID header and org_name path.

    Returns:
        LoadParsedResponse matching DocumentParseResponse format
    """
    start_time = time.time()

    # Construct paths
    parsed_path = _construct_parsed_path(
        request.org_name,
        request.folder_name,
        request.document_name
    )
    original_path = _construct_original_path(
        request.org_name,
        request.folder_name,
        request.document_name
    )

    logger.info(f"Loading parsed content: {parsed_path}")

    try:
        storage = get_storage()

        # Check if parsed file exists
        exists = await storage.exists(parsed_path, use_prefix=False)
        if not exists:
            logger.warning(f"Parsed content not found: {parsed_path}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Parsed content not found: {parsed_path}"
            )

        # Read the parsed content
        content = await storage.read(parsed_path, use_prefix=False)

        if content is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Failed to read parsed content: {parsed_path}"
            )

        processing_time_ms = (time.time() - start_time) * 1000
        content_length = len(content)

        # Estimate page count from content (rough approximation)
        # Assuming ~3000 chars per page for markdown
        estimated_pages = max(1, content_length // 3000)

        logger.info(f"Successfully loaded {content_length} bytes from {parsed_path}")

        return LoadParsedResponse(
            success=True,
            storage_path=original_path,
            parsed_storage_path=parsed_path,
            parsed_content=content,
            parsing_metadata={
                "total_pages": estimated_pages,
                "has_headers": False,
                "has_footers": False,
                "content_length": content_length,
                "parsing_duration": processing_time_ms / 1000,  # Convert to seconds
                "source": "gcs_load",  # Indicate this was loaded, not parsed
            },
            gcs_metadata={
                "size": content_length,
                "content_type": "text/markdown",
                "created": "",
                "updated": "",
            },
            file_info={
                "original_size": 0,  # Unknown when loading
                "parsed_size": content_length,
                "file_type": "markdown",
                "content_type": "text/markdown",
            },
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to load parsed content: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load parsed content: {str(e)}"
        )


@router.get(
    "/check-parsed",
    response_model=CheckParsedExistsResponse,
    operation_id="checkParsedExists",
    summary="Check if parsed content exists in GCS",
)
async def check_parsed_exists(
    org_name: str = Query(..., description="Organization name"),
    folder_name: str = Query(..., description="Folder name"),
    document_name: str = Query(..., description="Document name"),
    org_id: str = Depends(get_org_id),
) -> CheckParsedExistsResponse:
    """
    Check if parsed content exists for a document.

    This is a lightweight check that can be used to determine
    whether to show the "Load Parsed" button.

    Returns:
        CheckParsedExistsResponse with exists boolean and path
    """
    parsed_path = _construct_parsed_path(org_name, folder_name, document_name)

    try:
        storage = get_storage()
        exists = await storage.exists(parsed_path, use_prefix=False)

        return CheckParsedExistsResponse(
            exists=exists,
            path=parsed_path,
        )
    except Exception as e:
        logger.error(f"Error checking parsed existence: {e}")
        return CheckParsedExistsResponse(
            exists=False,
            path=parsed_path,
            error=str(e),
        )
