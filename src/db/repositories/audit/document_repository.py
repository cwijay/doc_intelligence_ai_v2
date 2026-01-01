"""Document repository - CRUD operations for document records.

Multi-tenancy: All operations are scoped by organization_id for tenant isolation.
"""

import hashlib
import logging
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional, List, Dict, Any

from sqlalchemy import select, and_, desc
from sqlalchemy.dialects.postgresql import insert

from ...models import Document
from ...connection import db
from ...utils import with_db_retry

logger = logging.getLogger(__name__)


# =============================================================================
# FILE HASH COMPUTATION
# =============================================================================


@lru_cache(maxsize=1000)
def _compute_file_hash(file_path: str, mtime: float) -> str:
    """Internal cached hash computation keyed by path and modification time."""
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def get_file_hash(file_path: str) -> str:
    """
    Compute SHA-256 hash of file content with caching.

    Uses LRU cache keyed by file path and modification time,
    so unchanged files don't need re-hashing.

    Args:
        file_path: Path to the file

    Returns:
        Hexadecimal hash string (64 characters)
    """
    mtime = os.path.getmtime(file_path)
    return _compute_file_hash(file_path, mtime)


# =============================================================================
# DOCUMENT REGISTRATION
# =============================================================================


@with_db_retry
async def register_document(file_path: str) -> str:
    """
    Register document if not exists, return file_hash.

    Uses PostgreSQL upsert (INSERT ... ON CONFLICT DO NOTHING) for
    atomic deduplication.

    Args:
        file_path: Path to the document

    Returns:
        SHA-256 hash of the file content
    """
    file_hash = get_file_hash(file_path)
    path = Path(file_path)

    async with db.session() as session:
        stmt = (
            insert(Document)
            .values(
                file_hash=file_hash,
                file_path=file_path,
                file_name=path.name,
                file_size=path.stat().st_size,
                created_at=datetime.utcnow(),
            )
            .on_conflict_do_nothing(index_elements=["file_hash"])
        )

        result = await session.execute(stmt)

        if result.rowcount > 0:
            logger.debug(f"Registered new document: {path.name} ({file_hash[:8]}...)")

    return file_hash


@with_db_retry
async def register_uploaded_document(
    file_path: str,
    file_name: str,
    file_size: int,
    organization_id: str,
    folder_name: Optional[str] = None,
) -> str:
    """
    Register a new uploaded document with status='uploaded'.

    Checks for existing active document with same (org_id, folder_id, filename)
    to prevent duplicates. If exists, updates it. Otherwise creates new.

    Multi-tenancy: Scoped by organization_id.

    Args:
        file_path: GCS URI or local path (unique identifier)
        file_name: Original filename for display
        file_size: File size in bytes
        organization_id: Organization ID for tenant isolation
        folder_name: Optional folder context (used as folder_id)

    Returns:
        File hash (SHA-256 if computed, or hash of file_path as fallback)
    """
    # Normalize storage_path to always include gs:// prefix
    normalized_path = file_path
    if not file_path.startswith("gs://"):
        try:
            from src.storage import get_storage
            storage = get_storage()
            normalized_path = f"gs://{storage.bucket_name}/{file_path}"
        except Exception as e:
            logger.warning(f"Could not normalize path, using original: {e}")
            normalized_path = file_path

    # Generate hash from normalized path
    file_hash = hashlib.sha256(normalized_path.encode()).hexdigest()

    # Infer file type from extension
    ext = Path(file_path).suffix.lower().lstrip(".")
    file_type = ext if ext else "unknown"

    async with db.session() as session:
        if session is None:
            return file_hash

        now = datetime.utcnow()

        # Check for existing active document with same org + folder + filename
        existing_stmt = (
            select(Document)
            .where(
                and_(
                    Document.organization_id == organization_id,
                    Document.folder_id == folder_name,
                    Document.filename == file_name,
                    Document.is_active == True,
                )
            )
            .limit(1)
        )
        result = await session.execute(existing_stmt)
        existing = result.scalar_one_or_none()

        if existing:
            # Update existing record instead of creating duplicate
            existing.storage_path = normalized_path
            existing.file_size = file_size
            existing.file_hash = file_hash
            existing.updated_at = now
            # Don't change status if already parsed
            if existing.status != "parsed":
                existing.status = "uploaded"
            logger.info(
                f"Updated existing document: {file_name} in folder={folder_name} "
                f"org={organization_id}"
            )
            return existing.file_hash

        # No existing - create new record
        stmt = (
            insert(Document)
            .values(
                file_hash=file_hash,
                storage_path=normalized_path,
                filename=file_name,
                original_filename=file_name,
                file_type=file_type,
                file_size=file_size,
                organization_id=organization_id,
                folder_id=folder_name,
                status="uploaded",
                uploaded_by="system",
                is_active=True,
                created_at=now,
                updated_at=now,
            )
            .on_conflict_do_update(
                index_elements=["file_hash"],
                set_={
                    "filename": file_name,
                    "file_size": file_size,
                    "folder_id": folder_name,
                    "status": "uploaded",
                    "updated_at": now,
                }
            )
        )

        await session.execute(stmt)
        logger.debug(f"Registered uploaded document: {file_name} at {normalized_path} org={organization_id}")

    return file_hash


@with_db_retry
async def update_document_status(
    file_path: str,
    status: str,
    organization_id: str,
    parsed_path: Optional[str] = None,
) -> bool:
    """
    Update document status after parsing.

    Args:
        file_path: GCS URI or local path (unique lookup key)
        status: New status ('parsed' or 'failed')
        organization_id: Organization ID for tenant isolation
        parsed_path: Path to parsed .md file (for status='parsed')

    Returns:
        True if document was found and updated, False otherwise
    """
    async with db.session() as session:
        if session is None:
            return False

        stmt = select(Document).where(
            and_(
                Document.storage_path == file_path,
                Document.organization_id == organization_id,
            )
        )
        result = await session.execute(stmt)
        doc = result.scalar_one_or_none()

        if not doc:
            logger.warning(f"Document not found for status update: {file_path} org={organization_id}")
            return False

        doc.status = status
        if status == "parsed":
            doc.parsed_path = parsed_path
            doc.parsed_at = datetime.utcnow()

        logger.info(f"Updated document status to '{status}': {file_path} org={organization_id}")
        return True


@with_db_retry
async def register_or_update_parsed_document(
    storage_path: str,
    filename: str,
    organization_id: str,
    parsed_path: str,
    file_size: Optional[int] = None,
    folder_id: Optional[str] = None,
) -> bool:
    """
    Update existing document to parsed status, or create new if not found.

    Lookup strategy: Find existing active document by (org_id, folder_id, filename).
    This prevents duplicate records for the same file.

    Multi-tenancy: Scoped by organization_id.

    Args:
        storage_path: GCS URI of the original document
        filename: Filename for display
        organization_id: Organization ID for tenant isolation
        parsed_path: GCS path to parsed .md file
        file_size: Optional file size in bytes
        folder_id: Folder name (not UUID) for organization

    Returns:
        True if document was updated/created, False on error
    """
    # Normalize storage_path to always include gs:// prefix
    normalized_path = storage_path
    if not storage_path.startswith("gs://"):
        try:
            from src.storage import get_storage
            storage = get_storage()
            normalized_path = f"gs://{storage.bucket_name}/{storage_path}"
        except Exception as e:
            logger.warning(f"Could not normalize path, using original: {e}")
            normalized_path = storage_path

    async with db.session() as session:
        if session is None:
            return False

        now = datetime.utcnow()

        # Primary lookup: by org + folder + filename + is_active
        stmt = (
            select(Document)
            .where(
                and_(
                    Document.organization_id == organization_id,
                    Document.folder_id == folder_id,
                    Document.filename == filename,
                    Document.is_active == True,
                )
            )
            .limit(1)
        )
        result = await session.execute(stmt)
        doc = result.scalar_one_or_none()

        # Fallback: try by org + filename only (for backwards compatibility)
        if not doc:
            stmt = (
                select(Document)
                .where(
                    and_(
                        Document.organization_id == organization_id,
                        Document.filename == filename,
                        Document.is_active == True,
                    )
                )
                .limit(1)
            )
            result = await session.execute(stmt)
            doc = result.scalar_one_or_none()
            if doc:
                logger.info(f"Found document by filename fallback: {filename}")

        if doc:
            # UPDATE existing record
            doc.status = "parsed"
            doc.parsed_path = parsed_path
            doc.parsed_at = now
            doc.updated_at = now
            # Normalize storage_path if different
            if doc.storage_path != normalized_path:
                doc.storage_path = normalized_path
            # Update folder_id if provided and different
            if folder_id and doc.folder_id != folder_id:
                doc.folder_id = folder_id
            logger.info(
                f"Updated existing document to parsed: {filename} "
                f"folder={folder_id} parsed_path={parsed_path} org={organization_id}"
            )
        else:
            # INSERT new record (fallback for documents not pre-registered)
            file_hash = hashlib.sha256(normalized_path.encode()).hexdigest()
            ext = Path(storage_path).suffix.lower().lstrip(".")
            file_type = ext if ext else "unknown"

            new_doc = Document(
                file_hash=file_hash,
                storage_path=normalized_path,
                filename=filename,
                original_filename=filename,
                file_type=file_type,
                file_size=file_size or 0,
                organization_id=organization_id,
                folder_id=folder_id,
                status="parsed",
                parsed_path=parsed_path,
                parsed_at=now,
                uploaded_by="system",
                is_active=True,
                created_at=now,
                updated_at=now,
            )
            session.add(new_doc)
            logger.info(
                f"Created new parsed document: {filename} at {normalized_path} "
                f"folder={folder_id} parsed_path={parsed_path} org={organization_id}"
            )

        return True


# =============================================================================
# DOCUMENT QUERIES
# =============================================================================


@with_db_retry
async def get_document_by_path(
    file_path: str,
    organization_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Get document by file_path (GCS URI).

    Args:
        file_path: GCS URI or local path (unique lookup key)
        organization_id: Organization ID for tenant isolation

    Returns:
        Document data dictionary or None if not found
    """
    async with db.session() as session:
        if session is None:
            return None

        stmt = select(Document).where(
            and_(
                Document.storage_path == file_path,
                Document.organization_id == organization_id,
            )
        )
        result = await session.execute(stmt)
        doc = result.scalar_one_or_none()

        if doc:
            return {
                "file_hash": doc.file_hash,
                "storage_path": doc.storage_path,
                "filename": doc.filename,
                "file_size": doc.file_size,
                "organization_id": doc.organization_id,
                "folder_id": doc.folder_id,
                "status": doc.status,
                "parsed_path": doc.parsed_path,
                "parsed_at": doc.parsed_at,
                "created_at": doc.created_at,
            }
        return None


@with_db_retry
async def get_document_by_name(file_name: str) -> Optional[Dict[str, Any]]:
    """
    Find document by file name.

    Args:
        file_name: Name of the document file

    Returns:
        Document data with hash, or None if not found
    """
    async with db.session() as session:
        stmt = select(Document).where(Document.filename == file_name).limit(1)
        result = await session.execute(stmt)
        doc = result.scalar_one_or_none()

        if doc:
            return {
                "file_hash": doc.file_hash,
                "storage_path": doc.storage_path,
                "filename": doc.filename,
                "file_size": doc.file_size,
                "created_at": doc.created_at,
            }
        return None


@with_db_retry
async def list_documents_by_status(
    organization_id: str,
    status: Optional[str] = None,
    folder_id: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    List documents for an organization, optionally filtered by status.

    Multi-tenancy: Scoped by organization_id.

    Args:
        organization_id: Organization ID for tenant isolation
        status: Optional status filter ('uploaded', 'parsed', 'failed')
        folder_id: Optional folder ID filter
        limit: Maximum number of results

    Returns:
        List of document dictionaries
    """
    async with db.session() as session:
        if session is None:
            return []

        where_clauses = [Document.organization_id == organization_id]
        if status:
            where_clauses.append(Document.status == status)
        if folder_id:
            where_clauses.append(Document.folder_id == folder_id)

        stmt = (
            select(Document)
            .where(and_(*where_clauses))
            .order_by(desc(Document.created_at))
            .limit(limit)
        )
        result = await session.execute(stmt)
        docs = result.scalars().all()

        return [
            {
                "file_hash": doc.file_hash,
                "storage_path": doc.storage_path,
                "filename": doc.filename,
                "file_size": doc.file_size,
                "organization_id": doc.organization_id,
                "folder_id": doc.folder_id,
                "status": doc.status,
                "parsed_path": doc.parsed_path,
                "parsed_at": doc.parsed_at,
                "created_at": doc.created_at,
            }
            for doc in docs
        ]
