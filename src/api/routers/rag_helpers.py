"""Helper functions for RAG router.

This module contains validation, content hashing, and store management
utilities extracted from the main rag.py router for better organization.
"""

import hashlib
import logging
from typing import Optional

from fastapi import HTTPException

from src.db.repositories import rag_repository

logger = logging.getLogger(__name__)


async def validate_store_ownership(store_id: str, org_id: str) -> dict:
    """Validate store exists and belongs to the requesting organization.

    Args:
        store_id: The store ID to validate
        org_id: The requesting organization's ID

    Returns:
        Store info dict if valid

    Raises:
        HTTPException: 404 if store not found, 403 if belongs to another org
    """
    # First try to get store without org filter to distinguish 404 from 403
    store_info = await rag_repository.get_store_by_id(store_id)

    if not store_info:
        raise HTTPException(
            status_code=404,
            detail=f"Store not found: {store_id}"
        )

    store_org_id = store_info.get("organization_id")
    if store_org_id and store_org_id != org_id:
        raise HTTPException(
            status_code=403,
            detail=f"Store '{store_id}' belongs to another organization"
        )

    return store_info


async def validate_folder_ownership(folder_id: str, org_id: str) -> dict:
    """Validate folder exists and belongs to the requesting organization.

    Args:
        folder_id: The folder ID to validate
        org_id: The requesting organization's ID

    Returns:
        Folder info dict if valid

    Raises:
        HTTPException: 404 if folder not found, 403 if belongs to another org
    """
    # First try to get folder without org filter to distinguish 404 from 403
    folder_info = await rag_repository.get_folder_by_id(folder_id)

    if not folder_info:
        raise HTTPException(
            status_code=404,
            detail=f"Folder not found: {folder_id}"
        )

    folder_org_id = folder_info.get("organization_id")
    if folder_org_id and folder_org_id != org_id:
        raise HTTPException(
            status_code=403,
            detail=f"Folder '{folder_id}' belongs to another organization"
        )

    return folder_info


def compute_content_hash(file_path: str) -> str:
    """Compute SHA-256 hash of local file content.

    Args:
        file_path: Path to local file

    Returns:
        Hexadecimal hash string
    """
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


async def compute_gcs_content_hash(gcs_path: str) -> Optional[str]:
    """Compute SHA-256 hash of GCS file content.

    Args:
        gcs_path: GCS URI (gs://bucket/path)

    Returns:
        Hexadecimal hash string or None if file not found
    """
    try:
        from src.storage import get_storage

        storage = get_storage()
        content = await storage.read(gcs_path)
        if content:
            # Handle both string and bytes content
            if isinstance(content, str):
                content = content.encode('utf-8')
            return hashlib.sha256(content).hexdigest()
    except Exception as e:
        logger.warning(f"Failed to compute hash for {gcs_path}: {e}")
    return None


async def get_or_create_org_store(org_id: str, org_name: str) -> dict:
    """Get or create org-specific Gemini File Search store.

    Uses naming convention: <org_name>_file_search_store

    Args:
        org_id: Organization ID (from header)
        org_name: Organization name (from request body)

    Returns:
        Store info dict with 'id', 'gemini_store_id', 'display_name', etc.
    """
    from src.rag.gemini_file_store import (
        generate_store_display_name,
        get_or_create_store_by_org_name,
    )

    display_name = generate_store_display_name(org_name)

    # First check PostgreSQL for existing store by display_name
    existing_store = await rag_repository.get_store_by_display_name(
        display_name,
        organization_id=org_id
    )

    if existing_store:
        logger.info(f"Found existing store in DB: {existing_store['id']} ({display_name})")
        return existing_store

    # Check/create in Gemini
    gemini_store, is_new = get_or_create_store_by_org_name(org_name)

    if is_new:
        # Register new store in PostgreSQL
        store_data = await rag_repository.create_store(
            organization_id=org_id,
            gemini_store_id=gemini_store.name,
            display_name=display_name,
            description=f"File search store for {org_name}",
        )
        logger.info(f"Created new store in DB: {store_data['id']} ({display_name})")
        return store_data
    else:
        # Gemini store exists but may not be in our DB - get or create record
        store_data = await rag_repository.get_or_create_store(
            organization_id=org_id,
            gemini_store_id=gemini_store.name,
            display_name=display_name,
            description=f"File search store for {org_name}",
        )
        logger.info(f"Linked existing Gemini store to DB: {store_data['id']} ({display_name})")
        return store_data
