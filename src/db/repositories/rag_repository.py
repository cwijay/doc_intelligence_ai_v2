"""
RAG repository - PostgreSQL implementation for file stores and folders.

Multi-tenancy: All operations are scoped by organization_id for tenant isolation.

Provides CRUD operations for:
- FileSearchStore: Gemini file store registry (one per organization)
- DocumentFolder: Document folder hierarchy
"""

import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import uuid4

from sqlalchemy import select, and_, desc, update, delete
from sqlalchemy.dialects.postgresql import insert

from ..models import FileSearchStore, DocumentFolder
from ..connection import db
from ..utils import with_db_retry

logger = logging.getLogger(__name__)


def _is_db_enabled() -> bool:
    """Check if database is enabled."""
    return db.config.enabled


# =============================================================================
# FILE SEARCH STORE OPERATIONS
# =============================================================================


@with_db_retry
async def create_store(
    organization_id: str,
    gemini_store_id: str,
    display_name: str,
    description: Optional[str] = None,
    gcp_project: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Create a new file search store for an organization.

    Multi-tenancy: One store per organization (enforced by unique constraint).

    Args:
        organization_id: Organization ID (one store per org)
        gemini_store_id: Gemini API store identifier
        display_name: Human-readable store name
        description: Optional description
        gcp_project: GCP project for multi-project scaling

    Returns:
        Store data dict if created, None if database disabled or org already has a store
    """
    async with db.session() as session:
        if session is None:
            return None

        # Check if org already has a store
        existing = await session.execute(
            select(FileSearchStore).where(FileSearchStore.organization_id == organization_id)
        )
        if existing.scalar_one_or_none():
            logger.warning(f"Organization {organization_id} already has a store")
            return None

        store = FileSearchStore(
            id=str(uuid4()),
            organization_id=organization_id,
            gemini_store_id=gemini_store_id,
            display_name=display_name,
            description=description,
            gcp_project=gcp_project,
            status="active",
            active_documents_count=0,
            total_size_bytes=0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        session.add(store)
        await session.flush()

        logger.info(f"Created store {store.id} for org {organization_id}")

        return {
            "id": store.id,
            "organization_id": store.organization_id,
            "gemini_store_id": store.gemini_store_id,
            "display_name": store.display_name,
            "description": store.description,
            "gcp_project": store.gcp_project,
            "status": store.status,
            "active_documents_count": store.active_documents_count,
            "total_size_bytes": store.total_size_bytes,
            "created_at": store.created_at,
            "updated_at": store.updated_at,
        }


@with_db_retry
async def get_store_by_org(organization_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the file search store for an organization.

    Multi-tenancy: Returns only the store belonging to the organization.

    Args:
        organization_id: Organization ID

    Returns:
        Store data dict if found, None otherwise
    """
    async with db.session() as session:
        if session is None:
            return None

        result = await session.execute(
            select(FileSearchStore).where(FileSearchStore.organization_id == organization_id)
        )
        store = result.scalar_one_or_none()

        if store:
            return {
                "id": store.id,
                "organization_id": store.organization_id,
                "gemini_store_id": store.gemini_store_id,
                "display_name": store.display_name,
                "description": store.description,
                "gcp_project": store.gcp_project,
                "status": store.status,
                "active_documents_count": store.active_documents_count,
                "total_size_bytes": store.total_size_bytes,
                "created_at": store.created_at,
                "updated_at": store.updated_at,
            }
        return None


@with_db_retry
async def get_store_by_id(
    store_id: str,
    organization_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get a file search store by ID.

    Multi-tenancy: Optionally validates ownership.

    Args:
        store_id: Store ID
        organization_id: Organization ID for ownership validation

    Returns:
        Store data dict if found, None otherwise
    """
    async with db.session() as session:
        if session is None:
            return None

        where_clauses = [FileSearchStore.id == store_id]
        if organization_id:
            where_clauses.append(FileSearchStore.organization_id == organization_id)

        result = await session.execute(
            select(FileSearchStore).where(and_(*where_clauses))
        )
        store = result.scalar_one_or_none()

        if store:
            return {
                "id": store.id,
                "organization_id": store.organization_id,
                "gemini_store_id": store.gemini_store_id,
                "display_name": store.display_name,
                "description": store.description,
                "gcp_project": store.gcp_project,
                "status": store.status,
                "active_documents_count": store.active_documents_count,
                "total_size_bytes": store.total_size_bytes,
                "created_at": store.created_at,
                "updated_at": store.updated_at,
            }
        return None


@with_db_retry
async def get_store_by_display_name(
    display_name: str,
    organization_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get a file search store by its display name.

    Used for auto-store creation to check if a store with the org-based
    naming convention already exists.

    Args:
        display_name: Store display name (e.g., "Acme_Corp_file_search_store")
        organization_id: Organization ID for ownership validation

    Returns:
        Store data dict if found, None otherwise
    """
    async with db.session() as session:
        if session is None:
            return None

        where_clauses = [FileSearchStore.display_name == display_name]
        if organization_id:
            where_clauses.append(FileSearchStore.organization_id == organization_id)

        result = await session.execute(
            select(FileSearchStore).where(and_(*where_clauses))
        )
        store = result.scalar_one_or_none()

        if store:
            return {
                "id": store.id,
                "organization_id": store.organization_id,
                "gemini_store_id": store.gemini_store_id,
                "display_name": store.display_name,
                "description": store.description,
                "gcp_project": store.gcp_project,
                "status": store.status,
                "active_documents_count": store.active_documents_count,
                "total_size_bytes": store.total_size_bytes,
                "created_at": store.created_at,
                "updated_at": store.updated_at,
            }
        return None


@with_db_retry
async def list_stores(organization_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List file search stores.

    Multi-tenancy: Filters by organization_id when provided.

    Args:
        organization_id: Organization ID for filtering

    Returns:
        List of store data dicts
    """
    async with db.session() as session:
        if session is None:
            return []

        stmt = select(FileSearchStore)
        if organization_id:
            stmt = stmt.where(FileSearchStore.organization_id == organization_id)
        stmt = stmt.order_by(desc(FileSearchStore.created_at))

        result = await session.execute(stmt)
        stores = result.scalars().all()

        return [
            {
                "id": store.id,
                "organization_id": store.organization_id,
                "gemini_store_id": store.gemini_store_id,
                "display_name": store.display_name,
                "description": store.description,
                "gcp_project": store.gcp_project,
                "status": store.status,
                "active_documents_count": store.active_documents_count,
                "total_size_bytes": store.total_size_bytes,
                "created_at": store.created_at,
                "updated_at": store.updated_at,
            }
            for store in stores
        ]


@with_db_retry
async def update_store_stats(
    store_id: str,
    documents_delta: int = 0,
    size_delta: int = 0,
) -> bool:
    """
    Update store statistics (document count and size).

    Args:
        store_id: Store ID
        documents_delta: Change in document count (+/-)
        size_delta: Change in total size bytes (+/-)

    Returns:
        True if updated, False otherwise
    """
    async with db.session() as session:
        if session is None:
            return False

        result = await session.execute(
            select(FileSearchStore).where(FileSearchStore.id == store_id)
        )
        store = result.scalar_one_or_none()

        if store:
            store.active_documents_count = max(0, store.active_documents_count + documents_delta)
            store.total_size_bytes = max(0, store.total_size_bytes + size_delta)
            store.updated_at = datetime.utcnow()
            return True
        return False


@with_db_retry
async def delete_store(
    store_id: str,
    organization_id: Optional[str] = None,
) -> bool:
    """
    Delete a file search store.

    Multi-tenancy: Validates ownership when organization_id provided.

    Args:
        store_id: Store ID
        organization_id: Organization ID for ownership validation

    Returns:
        True if deleted, False otherwise
    """
    async with db.session() as session:
        if session is None:
            return False

        where_clauses = [FileSearchStore.id == store_id]
        if organization_id:
            where_clauses.append(FileSearchStore.organization_id == organization_id)

        result = await session.execute(
            delete(FileSearchStore).where(and_(*where_clauses))
        )

        if result.rowcount > 0:
            logger.info(f"Deleted store {store_id}")
            return True
        return False


# =============================================================================
# DOCUMENT FOLDER OPERATIONS
# =============================================================================


@with_db_retry
async def create_folder(
    organization_id: str,
    store_id: str,
    folder_name: str,
    description: Optional[str] = None,
    parent_folder_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Create a new document folder.

    Multi-tenancy: Folder is associated with the organization.

    Args:
        organization_id: Organization ID
        store_id: File search store ID
        folder_name: Folder name
        description: Optional description
        parent_folder_id: Parent folder ID for nested structure

    Returns:
        Folder data dict if created, None otherwise
    """
    async with db.session() as session:
        if session is None:
            return None

        # Check for duplicate folder name within parent
        existing = await session.execute(
            select(DocumentFolder).where(
                and_(
                    DocumentFolder.organization_id == organization_id,
                    DocumentFolder.folder_name == folder_name,
                    DocumentFolder.parent_folder_id == parent_folder_id,
                )
            )
        )
        if existing.scalar_one_or_none():
            logger.warning(f"Folder '{folder_name}' already exists in this location")
            return None

        folder = DocumentFolder(
            id=str(uuid4()),
            organization_id=organization_id,
            store_id=store_id,
            folder_name=folder_name,
            description=description,
            parent_folder_id=parent_folder_id,
            document_count=0,
            total_size_bytes=0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        session.add(folder)
        await session.flush()

        logger.info(f"Created folder '{folder_name}' ({folder.id}) for org {organization_id}")

        return {
            "id": folder.id,
            "organization_id": folder.organization_id,
            "store_id": folder.store_id,
            "folder_name": folder.folder_name,
            "description": folder.description,
            "parent_folder_id": folder.parent_folder_id,
            "document_count": folder.document_count,
            "total_size_bytes": folder.total_size_bytes,
            "created_at": folder.created_at,
            "updated_at": folder.updated_at,
        }


@with_db_retry
async def get_folder_by_id(
    folder_id: str,
    organization_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get a folder by ID.

    Multi-tenancy: Optionally validates ownership.

    Args:
        folder_id: Folder ID
        organization_id: Organization ID for ownership validation

    Returns:
        Folder data dict if found, None otherwise
    """
    async with db.session() as session:
        if session is None:
            return None

        where_clauses = [DocumentFolder.id == folder_id]
        if organization_id:
            where_clauses.append(DocumentFolder.organization_id == organization_id)

        result = await session.execute(
            select(DocumentFolder).where(and_(*where_clauses))
        )
        folder = result.scalar_one_or_none()

        if folder:
            return {
                "id": folder.id,
                "organization_id": folder.organization_id,
                "store_id": folder.store_id,
                "folder_name": folder.folder_name,
                "description": folder.description,
                "parent_folder_id": folder.parent_folder_id,
                "document_count": folder.document_count,
                "total_size_bytes": folder.total_size_bytes,
                "created_at": folder.created_at,
                "updated_at": folder.updated_at,
            }
        return None


@with_db_retry
async def get_folder_by_name(
    organization_id: str,
    folder_name: str,
    parent_folder_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get a folder by name within a parent folder.

    Args:
        organization_id: Organization ID
        folder_name: Folder name
        parent_folder_id: Parent folder ID (None for root folders)

    Returns:
        Folder data dict if found, None otherwise
    """
    async with db.session() as session:
        if session is None:
            return None

        result = await session.execute(
            select(DocumentFolder).where(
                and_(
                    DocumentFolder.organization_id == organization_id,
                    DocumentFolder.folder_name == folder_name,
                    DocumentFolder.parent_folder_id == parent_folder_id,
                )
            )
        )
        folder = result.scalar_one_or_none()

        if folder:
            return {
                "id": folder.id,
                "organization_id": folder.organization_id,
                "store_id": folder.store_id,
                "folder_name": folder.folder_name,
                "description": folder.description,
                "parent_folder_id": folder.parent_folder_id,
                "document_count": folder.document_count,
                "total_size_bytes": folder.total_size_bytes,
                "created_at": folder.created_at,
                "updated_at": folder.updated_at,
            }
        return None


@with_db_retry
async def list_folders(
    organization_id: str,
    parent_folder_id: Optional[str] = None,
    include_all: bool = False,
) -> List[Dict[str, Any]]:
    """
    List folders for an organization.

    Multi-tenancy: Returns only folders belonging to the organization.

    Args:
        organization_id: Organization ID
        parent_folder_id: Filter by parent folder (None for root folders)
        include_all: If True, return all folders regardless of parent

    Returns:
        List of folder data dicts
    """
    async with db.session() as session:
        if session is None:
            return []

        where_clauses = [DocumentFolder.organization_id == organization_id]

        if not include_all:
            if parent_folder_id is not None:
                where_clauses.append(DocumentFolder.parent_folder_id == parent_folder_id)
            else:
                where_clauses.append(DocumentFolder.parent_folder_id.is_(None))

        result = await session.execute(
            select(DocumentFolder)
            .where(and_(*where_clauses))
            .order_by(DocumentFolder.folder_name)
        )
        folders = result.scalars().all()

        return [
            {
                "id": folder.id,
                "organization_id": folder.organization_id,
                "store_id": folder.store_id,
                "folder_name": folder.folder_name,
                "description": folder.description,
                "parent_folder_id": folder.parent_folder_id,
                "document_count": folder.document_count,
                "total_size_bytes": folder.total_size_bytes,
                "created_at": folder.created_at,
                "updated_at": folder.updated_at,
            }
            for folder in folders
        ]


@with_db_retry
async def update_folder(
    folder_id: str,
    organization_id: str,
    folder_name: Optional[str] = None,
    description: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Update a folder.

    Multi-tenancy: Validates ownership.

    Args:
        folder_id: Folder ID
        organization_id: Organization ID for ownership validation
        folder_name: New folder name (optional)
        description: New description (optional)

    Returns:
        Updated folder data dict if successful, None otherwise
    """
    async with db.session() as session:
        if session is None:
            return None

        result = await session.execute(
            select(DocumentFolder).where(
                and_(
                    DocumentFolder.id == folder_id,
                    DocumentFolder.organization_id == organization_id,
                )
            )
        )
        folder = result.scalar_one_or_none()

        if not folder:
            return None

        if folder_name is not None:
            folder.folder_name = folder_name
        if description is not None:
            folder.description = description
        folder.updated_at = datetime.utcnow()

        return {
            "id": folder.id,
            "organization_id": folder.organization_id,
            "store_id": folder.store_id,
            "folder_name": folder.folder_name,
            "description": folder.description,
            "parent_folder_id": folder.parent_folder_id,
            "document_count": folder.document_count,
            "total_size_bytes": folder.total_size_bytes,
            "created_at": folder.created_at,
            "updated_at": folder.updated_at,
        }


@with_db_retry
async def update_folder_stats(
    folder_id: str,
    documents_delta: int = 0,
    size_delta: int = 0,
) -> bool:
    """
    Update folder statistics (document count and size).

    Args:
        folder_id: Folder ID
        documents_delta: Change in document count (+/-)
        size_delta: Change in total size bytes (+/-)

    Returns:
        True if updated, False otherwise
    """
    async with db.session() as session:
        if session is None:
            return False

        result = await session.execute(
            select(DocumentFolder).where(DocumentFolder.id == folder_id)
        )
        folder = result.scalar_one_or_none()

        if folder:
            folder.document_count = max(0, folder.document_count + documents_delta)
            folder.total_size_bytes = max(0, folder.total_size_bytes + size_delta)
            folder.updated_at = datetime.utcnow()
            return True
        return False


@with_db_retry
async def has_subfolders(folder_id: str) -> bool:
    """
    Check if a folder has subfolders.

    Args:
        folder_id: Folder ID

    Returns:
        True if folder has subfolders, False otherwise
    """
    async with db.session() as session:
        if session is None:
            return False

        result = await session.execute(
            select(DocumentFolder.id)
            .where(DocumentFolder.parent_folder_id == folder_id)
            .limit(1)
        )
        return result.scalar_one_or_none() is not None


@with_db_retry
async def delete_folder(
    folder_id: str,
    organization_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Delete a folder.

    Multi-tenancy: Validates ownership when organization_id provided.

    Args:
        folder_id: Folder ID
        organization_id: Organization ID for ownership validation

    Returns:
        Dict with success status and document_count that was deleted
    """
    async with db.session() as session:
        if session is None:
            return {"success": False, "document_count": 0}

        # Get folder info first
        where_clauses = [DocumentFolder.id == folder_id]
        if organization_id:
            where_clauses.append(DocumentFolder.organization_id == organization_id)

        result = await session.execute(
            select(DocumentFolder).where(and_(*where_clauses))
        )
        folder = result.scalar_one_or_none()

        if not folder:
            return {"success": False, "document_count": 0}

        document_count = folder.document_count

        # Delete the folder
        await session.execute(
            delete(DocumentFolder).where(DocumentFolder.id == folder_id)
        )

        logger.info(f"Deleted folder {folder_id} with {document_count} documents")

        return {"success": True, "document_count": document_count}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


@with_db_retry
async def get_or_create_store(
    organization_id: str,
    gemini_store_id: str,
    display_name: str,
    description: Optional[str] = None,
    gcp_project: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get existing store for org or create a new one.

    Multi-tenancy: One store per organization.

    Args:
        organization_id: Organization ID
        gemini_store_id: Gemini API store identifier (used if creating)
        display_name: Display name (used if creating)
        description: Description (used if creating)
        gcp_project: GCP project (used if creating)

    Returns:
        Store data dict
    """
    # Try to get existing store
    existing = await get_store_by_org(organization_id)
    if existing:
        return existing

    # Create new store
    return await create_store(
        organization_id=organization_id,
        gemini_store_id=gemini_store_id,
        display_name=display_name,
        description=description,
        gcp_project=gcp_project,
    )
