"""RAG (Retrieval-Augmented Generation) API endpoints.

Multi-tenancy: All endpoints are scoped by organization_id from request headers.
Stores are isolated per organization (one store per org).
Documents are organized into folders with metadata filtering.
PostgreSQL persistence via rag_repository.
"""

import logging
import os
from datetime import datetime
from pathlib import Path as PathLib
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query

from ..dependencies import get_org_id
from ..schemas.errors import STORE_ERROR_RESPONSES, FOLDER_ERROR_RESPONSES
from ..schemas.rag import (
    CreateStoreRequest,
    CreateStoreResponse,
    ListStoresResponse,
    UploadToStoreRequest,
    UploadToStoreResponse,
    SearchStoreRequest,
    SearchStoreResponse,
    ListStoreFilesResponse,
    DeleteStoreResponse,
    StoreInfo,
    StoreFileInfo,
    Citation,
    # Folder schemas
    CreateFolderRequest,
    CreateFolderResponse,
    ListFoldersResponse,
    GetFolderResponse,
    DeleteFolderResponse,
    FolderInfo,
)
from src.db.repositories import rag_repository
from .rag_helpers import (
    validate_store_ownership,
    validate_folder_ownership,
    compute_content_hash,
    compute_gcs_content_hash,
    get_or_create_org_store,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/stores",
    response_model=CreateStoreResponse,
    responses=STORE_ERROR_RESPONSES,
    operation_id="createStore",
    summary="Create Gemini File Search store",
)
async def create_store(
    request: CreateStoreRequest,
    org_id: str = Depends(get_org_id),
):
    """
    Create a new Gemini File Search store for semantic document retrieval.

    **One store per organization**: If the organization already has a store, returns the existing store info.

    **Multi-tenancy**: Store is associated with the requesting organization via X-Organization-ID header.
    """
    try:
        # Check if org already has a store
        existing_store = await rag_repository.get_store_by_org(org_id)
        if existing_store:
            return CreateStoreResponse(
                success=True,
                store_id=existing_store["id"],
                display_name=existing_store["display_name"],
                created_at=existing_store["created_at"],
            )

        from src.rag.gemini_file_store import create_file_search_store

        gemini_store = create_file_search_store(
            display_name=request.display_name,
        )

        if gemini_store and gemini_store.name:
            # Persist to PostgreSQL
            store_data = await rag_repository.create_store(
                organization_id=org_id,
                gemini_store_id=gemini_store.name,
                display_name=request.display_name,
                description=request.description,
            )

            if store_data:
                return CreateStoreResponse(
                    success=True,
                    store_id=store_data["id"],
                    display_name=request.display_name,
                    created_at=store_data["created_at"],
                )

        return CreateStoreResponse(
            success=False,
            error="Failed to create store"
        )

    except ImportError:
        return CreateStoreResponse(
            success=False,
            error="Gemini File Store module not available"
        )
    except Exception as e:
        logger.exception(f"Failed to create store: {e}")
        return CreateStoreResponse(
            success=False,
            error=str(e)
        )


@router.get(
    "/stores",
    response_model=ListStoresResponse,
    responses=STORE_ERROR_RESPONSES,
    operation_id="listStores",
    summary="List organization's file search stores",
)
async def list_stores(
    org_id: str = Depends(get_org_id),
):
    """
    List all file search stores for the organization.

    **Multi-tenancy**: Only returns stores belonging to the requesting organization.
    """
    try:
        # Multi-tenancy: Filter stores by organization from PostgreSQL
        store_list = await rag_repository.list_stores(organization_id=org_id)

        stores = [
            StoreInfo(
                store_id=store["id"],
                display_name=store["display_name"],
                description=store.get("description"),
                created_at=store["created_at"],
                file_count=store.get("active_documents_count", 0),
                status=store.get("status", "active"),
            )
            for store in store_list
        ]

        return ListStoresResponse(
            success=True,
            stores=stores,
            count=len(stores),
        )

    except Exception as e:
        logger.exception(f"Failed to list stores: {e}")
        return ListStoresResponse(
            success=False,
            error=str(e)
        )


@router.get(
    "/stores/{store_id}",
    response_model=CreateStoreResponse,
    responses=STORE_ERROR_RESPONSES,
    operation_id="getStore",
    summary="Get store details",
)
async def get_store(
    store_id: str = Path(..., description="Store ID"),
    org_id: str = Depends(get_org_id),
):
    """
    Get details of a specific file search store.

    **Multi-tenancy**: Validates store belongs to the requesting organization.
    """
    try:
        # Multi-tenancy: Validate ownership (async)
        info = await validate_store_ownership(store_id, org_id)

        return CreateStoreResponse(
            success=True,
            store_id=store_id,
            display_name=info["display_name"],
            created_at=info["created_at"],
        )

    except HTTPException:
        raise  # Re-raise HTTP exceptions (403, 404)
    except Exception as e:
        logger.exception(f"Failed to get store: {e}")
        return CreateStoreResponse(
            success=False,
            error=str(e)
        )


@router.delete(
    "/stores/{store_id}",
    response_model=DeleteStoreResponse,
    responses=STORE_ERROR_RESPONSES,
    operation_id="deleteStore",
    summary="Delete file search store",
)
async def delete_store(
    store_id: str = Path(..., description="Store ID"),
    org_id: str = Depends(get_org_id),
):
    """
    Delete a file search store and all its contents.

    **Warning**: This is a destructive operation. All documents in the store will be deleted.

    **Multi-tenancy**: Validates store belongs to the requesting organization before deletion.
    """
    try:
        # Multi-tenancy: Validate ownership before deletion (async)
        store_info = await validate_store_ownership(store_id, org_id)

        from src.rag.gemini_file_store import delete_store as gemini_delete_store

        # Delete from Gemini using the gemini_store_id
        gemini_store_id = store_info.get("gemini_store_id")
        if gemini_store_id:
            gemini_delete_store(gemini_store_id)

        # Delete from PostgreSQL
        await rag_repository.delete_store(store_id, organization_id=org_id)

        return DeleteStoreResponse(
            success=True,
            store_id=store_id,
            message="Store deleted successfully"
        )

    except HTTPException:
        raise  # Re-raise HTTP exceptions (403, 404)
    except ImportError:
        return DeleteStoreResponse(
            success=False,
            store_id=store_id,
            error="Gemini File Store module not available"
        )
    except Exception as e:
        logger.exception(f"Failed to delete store: {e}")
        return DeleteStoreResponse(
            success=False,
            store_id=store_id,
            error=str(e)
        )


@router.post(
    "/stores/{store_id}/upload",
    response_model=UploadToStoreResponse,
    responses=STORE_ERROR_RESPONSES,
    operation_id="uploadToStore",
    summary="Upload files to store",
)
async def upload_to_store(
    request: UploadToStoreRequest,
    store_id: str = Path(..., description="Store ID or 'auto' for org-based auto-creation"),
    org_id: str = Depends(get_org_id),
):
    """
    Upload files to a file search store for semantic retrieval.

    **Auto-creation**: Use store_id='auto' to automatically create or reuse
    an org-specific store named '<org_name>_file_search_store'.

    **Enhanced metadata**: Each uploaded document includes:
    - Original document name, org_name, folder_name
    - Parse date, parser_version, original_file_extension
    - Original GCS path, parsed GCS path
    - Content hash (SHA-256) for deduplication

    **Folder organization**: Optionally specify a folder to organize uploaded documents.

    **Multi-tenancy**: Validates store belongs to the requesting organization.
    """
    actual_store_id = store_id  # Track for response

    try:
        # Handle auto store creation/retrieval
        if store_id == "auto":
            if not request.org_name:
                raise HTTPException(
                    status_code=400,
                    detail="org_name is required when using store_id='auto'"
                )
            store_info = await get_or_create_org_store(org_id, request.org_name)
            actual_store_id = store_info["id"]
        else:
            # Multi-tenancy: Validate ownership (async)
            store_info = await validate_store_ownership(store_id, org_id)

        # Validate folder ownership if folder_id provided
        folder_name = request.folder_name
        if request.folder_id:
            folder_info = await validate_folder_ownership(request.folder_id, org_id)
            folder_name = folder_name or folder_info.get("folder_name")

        from src.rag.gemini_file_store import upload_file
        from google import genai

        # Get the Gemini store object
        gemini_store_id = store_info.get("gemini_store_id")
        gemini_store = None
        if gemini_store_id:
            client = genai.Client()
            for store in client.file_search_stores.list():
                if store.name == gemini_store_id:
                    gemini_store = store
                    break

        uploaded_files = []
        errors = []
        total_size = 0

        for idx, file_path in enumerate(request.file_paths):
            try:
                if gemini_store:
                    # Compute content hash
                    content_hash = None
                    if file_path.startswith("gs://"):
                        content_hash = await compute_gcs_content_hash(file_path)
                    elif os.path.exists(file_path):
                        content_hash = compute_content_hash(file_path)

                    # Get original GCS path if provided
                    original_gcs_path = None
                    if request.original_gcs_paths and idx < len(request.original_gcs_paths):
                        original_gcs_path = request.original_gcs_paths[idx]

                    # Extract original file extension from original path
                    original_ext = None
                    if original_gcs_path:
                        original_ext = PathLib(original_gcs_path).suffix

                    # Current timestamp for parse_date
                    parse_date = datetime.utcnow().isoformat()

                    upload_file(
                        gemini_store,
                        file_path,
                        organization_id=org_id,
                        folder_id=request.folder_id,
                        folder_name=folder_name,
                        # Enhanced metadata
                        org_name=request.org_name,
                        content_hash=content_hash,
                        original_gcs_path=original_gcs_path,
                        parsed_gcs_path=file_path,  # The file being uploaded is the parsed file
                        original_file_extension=original_ext,
                        original_file_size=None,  # Could be fetched if needed
                        parse_date=parse_date,
                        parser_version=request.parser_version,
                    )

                    file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                    total_size += file_size

                    uploaded_files.append(StoreFileInfo(
                        file_name=os.path.basename(file_path),
                        file_size_bytes=file_size,
                        upload_date=datetime.utcnow(),
                        status="ready",
                        # Enhanced metadata
                        content_hash=content_hash,
                        original_file_extension=original_ext,
                        original_gcs_path=original_gcs_path,
                        parsed_gcs_path=file_path,
                        parse_date=datetime.utcnow(),
                        parser_version=request.parser_version,
                        org_name=request.org_name,
                        folder_name=folder_name,
                    ))
            except Exception as upload_error:
                errors.append({"file": file_path, "error": str(upload_error)})

        # Update store stats in PostgreSQL
        if uploaded_files:
            await rag_repository.update_store_stats(
                store_id=actual_store_id,
                documents_delta=len(uploaded_files),
                size_delta=total_size,
            )

            # Update folder stats if folder specified
            if request.folder_id:
                await rag_repository.update_folder_stats(
                    folder_id=request.folder_id,
                    documents_delta=len(uploaded_files),
                    size_delta=total_size,
                )

        return UploadToStoreResponse(
            success=len(uploaded_files) > 0,
            store_id=actual_store_id,
            uploaded=len(uploaded_files),
            failed=len(errors),
            files=uploaded_files,
            errors=errors,
        )

    except HTTPException:
        raise  # Re-raise HTTP exceptions (403, 404)
    except ImportError:
        return UploadToStoreResponse(
            success=False,
            store_id=actual_store_id,
            error="Gemini File Store module not available"
        )
    except Exception as e:
        logger.exception(f"Failed to upload to store: {e}")
        return UploadToStoreResponse(
            success=False,
            store_id=actual_store_id,
            errors=[{"error": str(e)}],
        )


# NOTE: Search endpoint removed - use /api/v1/documents/chat instead
# The search functionality is now consolidated into the DocumentAgent


@router.get(
    "/stores/{store_id}/files",
    response_model=ListStoreFilesResponse,
    responses=STORE_ERROR_RESPONSES,
    operation_id="listStoreFiles",
    summary="List files in store",
)
async def list_store_files(
    store_id: str = Path(..., description="Store ID"),
    org_id: str = Depends(get_org_id),
):
    """
    List all files uploaded to a file search store.

    Returns file metadata including name, size, upload date, and processing status.

    **Multi-tenancy**: Validates store belongs to the requesting organization.
    """
    try:
        # Multi-tenancy: Validate ownership (async)
        store_info = await validate_store_ownership(store_id, org_id)

        from src.rag.gemini_file_store import list_documents
        from google import genai

        # Get the Gemini store object
        gemini_store_id = store_info.get("gemini_store_id")
        gemini_store = None
        if gemini_store_id:
            client = genai.Client()
            for store in client.file_search_stores.list():
                if store.name == gemini_store_id:
                    gemini_store = store
                    break

        files = []
        if gemini_store:
            documents = list_documents(gemini_store)
            for doc in documents:
                # Extract metadata from document
                file_name = doc.display_name if hasattr(doc, 'display_name') else None
                file_size = None
                upload_date = None

                if hasattr(doc, 'custom_metadata') and doc.custom_metadata:
                    for meta in doc.custom_metadata:
                        if meta.key == 'file_size' and hasattr(meta, 'numeric_value'):
                            file_size = int(meta.numeric_value) if meta.numeric_value else None
                        if meta.key == 'upload_date' and hasattr(meta, 'string_value'):
                            try:
                                upload_date = datetime.fromisoformat(meta.string_value)
                            except (ValueError, TypeError):
                                pass

                files.append(StoreFileInfo(
                    file_name=file_name,
                    file_size_bytes=file_size,
                    upload_date=upload_date,
                    status="ready",
                ))

        return ListStoreFilesResponse(
            success=True,
            store_id=store_id,
            files=files,
            count=len(files),
        )

    except HTTPException:
        raise  # Re-raise HTTP exceptions (403, 404)
    except ImportError:
        return ListStoreFilesResponse(
            success=False,
            store_id=store_id,
            error="Gemini File Store module not available"
        )
    except Exception as e:
        logger.exception(f"Failed to list store files: {e}")
        return ListStoreFilesResponse(
            success=False,
            store_id=store_id,
            error=str(e)
        )


# =============================================================================
# Folder Endpoints
# =============================================================================


@router.post(
    "/folders",
    response_model=CreateFolderResponse,
    responses=FOLDER_ERROR_RESPONSES,
    operation_id="createFolder",
    summary="Create document folder",
)
async def create_folder(
    request: CreateFolderRequest,
    org_id: str = Depends(get_org_id),
):
    """
    Create a new document folder for organizing files within a store.

    **Nested folders**: Use `parent_folder_id` to create subfolders.

    **Prerequisite**: Organization must have a store before creating folders.

    **Multi-tenancy**: Folder is associated with the requesting organization.
    """
    try:
        # Validate parent folder if specified (async)
        if request.parent_folder_id:
            await validate_folder_ownership(request.parent_folder_id, org_id)

        # Get the org's store (required for folder association)
        store = await rag_repository.get_store_by_org(org_id)
        if not store:
            return CreateFolderResponse(
                success=False,
                error="Organization must have a store before creating folders. Create a store first."
            )

        # Check for duplicate folder name within parent
        existing = await rag_repository.get_folder_by_name(
            organization_id=org_id,
            folder_name=request.folder_name,
            parent_folder_id=request.parent_folder_id,
        )
        if existing:
            return CreateFolderResponse(
                success=False,
                error=f"Folder '{request.folder_name}' already exists in this location"
            )

        # Create folder in PostgreSQL
        folder_data = await rag_repository.create_folder(
            organization_id=org_id,
            store_id=store["id"],
            folder_name=request.folder_name,
            description=request.description,
            parent_folder_id=request.parent_folder_id,
        )

        if not folder_data:
            return CreateFolderResponse(
                success=False,
                error="Failed to create folder"
            )

        folder_info = FolderInfo(
            id=folder_data["id"],
            folder_id=folder_data["id"],
            folder_name=folder_data["folder_name"],
            description=folder_data.get("description"),
            parent_folder_id=folder_data.get("parent_folder_id"),
            document_count=folder_data.get("document_count", 0),
            total_size_bytes=folder_data.get("total_size_bytes", 0),
            created_at=folder_data["created_at"],
            updated_at=folder_data["updated_at"],
        )

        logger.info(f"Created folder '{request.folder_name}' for org {org_id}")

        return CreateFolderResponse(
            success=True,
            folder=folder_info,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to create folder: {e}")
        return CreateFolderResponse(
            success=False,
            error=str(e)
        )


@router.get(
    "/folders",
    response_model=ListFoldersResponse,
    responses=FOLDER_ERROR_RESPONSES,
    operation_id="listFolders",
    summary="List document folders",
)
async def list_folders(
    parent_folder_id: Optional[str] = Query(None, description="Filter by parent folder (null for root folders)"),
    org_id: str = Depends(get_org_id),
):
    """
    List document folders for the organization.

    **Root folders**: Omit `parent_folder_id` to list top-level folders.

    **Subfolders**: Provide `parent_folder_id` to list children of a specific folder.

    **Multi-tenancy**: Only returns folders belonging to the requesting organization.
    """
    try:
        # Multi-tenancy: Filter folders by organization from PostgreSQL
        folder_list = await rag_repository.list_folders(
            organization_id=org_id,
            parent_folder_id=parent_folder_id,
        )

        folders = [
            FolderInfo(
                id=f["id"],
                folder_id=f["id"],
                folder_name=f["folder_name"],
                description=f.get("description"),
                parent_folder_id=f.get("parent_folder_id"),
                document_count=f.get("document_count", 0),
                total_size_bytes=f.get("total_size_bytes", 0),
                created_at=f["created_at"],
                updated_at=f["updated_at"],
            )
            for f in folder_list
        ]

        return ListFoldersResponse(
            success=True,
            folders=folders,
            count=len(folders),
        )

    except Exception as e:
        logger.exception(f"Failed to list folders: {e}")
        return ListFoldersResponse(
            success=False,
            error=str(e)
        )


@router.get(
    "/folders/{folder_id}",
    response_model=GetFolderResponse,
    responses=FOLDER_ERROR_RESPONSES,
    operation_id="getFolder",
    summary="Get folder details",
)
async def get_folder(
    folder_id: str = Path(..., description="Folder ID"),
    org_id: str = Depends(get_org_id),
):
    """
    Get details of a specific document folder.

    Returns folder metadata including document count and total size.

    **Multi-tenancy**: Validates folder belongs to the requesting organization.
    """
    try:
        # Multi-tenancy: Validate ownership (async)
        info = await validate_folder_ownership(folder_id, org_id)

        folder_info = FolderInfo(
            id=info["id"],
            folder_id=info["id"],
            folder_name=info["folder_name"],
            description=info.get("description"),
            parent_folder_id=info.get("parent_folder_id"),
            document_count=info.get("document_count", 0),
            total_size_bytes=info.get("total_size_bytes", 0),
            created_at=info["created_at"],
            updated_at=info["updated_at"],
        )

        return GetFolderResponse(
            success=True,
            folder=folder_info,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get folder: {e}")
        return GetFolderResponse(
            success=False,
            error=str(e)
        )


@router.delete(
    "/folders/{folder_id}",
    response_model=DeleteFolderResponse,
    responses=FOLDER_ERROR_RESPONSES,
    operation_id="deleteFolder",
    summary="Delete document folder",
)
async def delete_folder(
    folder_id: str = Path(..., description="Folder ID"),
    org_id: str = Depends(get_org_id),
):
    """
    Delete a document folder.

    **Restriction**: Cannot delete folders that have subfolders. Delete subfolders first.

    **Side effects**: Updates the store's document count statistics.

    **Multi-tenancy**: Validates folder belongs to the requesting organization before deletion.
    """
    try:
        # Multi-tenancy: Validate ownership before deletion (async)
        info = await validate_folder_ownership(folder_id, org_id)
        documents_to_delete = info.get("document_count", 0)
        size_to_delete = info.get("total_size_bytes", 0)
        store_id = info.get("store_id")

        # Check for subfolders using repository
        has_children = await rag_repository.has_subfolders(folder_id)
        if has_children:
            return DeleteFolderResponse(
                success=False,
                folder_id=folder_id,
                error="Cannot delete folder with subfolders. Delete subfolders first."
            )

        # Delete folder from PostgreSQL
        result = await rag_repository.delete_folder(folder_id, organization_id=org_id)

        if not result.get("success"):
            return DeleteFolderResponse(
                success=False,
                folder_id=folder_id,
                error="Failed to delete folder from database"
            )

        # Update store stats (subtract deleted folder's documents)
        if store_id and (documents_to_delete > 0 or size_to_delete > 0):
            await rag_repository.update_store_stats(
                store_id=store_id,
                documents_delta=-documents_to_delete,
                size_delta=-size_to_delete,
            )

        logger.info(f"Deleted folder {folder_id} for org {org_id}")

        return DeleteFolderResponse(
            success=True,
            folder_id=folder_id,
            message="Folder deleted successfully",
            documents_deleted=result.get("document_count", 0),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to delete folder: {e}")
        return DeleteFolderResponse(
            success=False,
            folder_id=folder_id,
            error=str(e)
        )
