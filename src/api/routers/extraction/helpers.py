"""Helper utilities for extraction endpoints.

Provides document loading and token mapping utilities.
"""

import logging
from typing import Optional

from fastapi import HTTPException, status

from src.api.schemas.extraction import TokenUsageSchema

logger = logging.getLogger(__name__)


async def load_document_content(parsed_file_path: str) -> str:
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


def map_token_usage(token_usage) -> Optional[TokenUsageSchema]:
    """Map agent token usage to API schema.

    Args:
        token_usage: Token usage from agent response

    Returns:
        TokenUsageSchema or None if no token usage
    """
    if not token_usage:
        return None

    return TokenUsageSchema(
        prompt_tokens=token_usage.prompt_tokens,
        completion_tokens=token_usage.completion_tokens,
        total_tokens=token_usage.total_tokens,
        estimated_cost_usd=token_usage.estimated_cost_usd
    )


async def get_organization_name_safe(org_id: str) -> str:
    """Safely get organization name, falling back to org_id.

    Args:
        org_id: Organization UUID

    Returns:
        Organization name or org_id as fallback
    """
    try:
        from src.db.repositories.extraction_repository import get_organization_name
        org_name = await get_organization_name(org_id)
        return org_name if org_name else org_id
    except Exception as e:
        logger.warning(f"Failed to get org name for {org_id}: {e}")
        return org_id
