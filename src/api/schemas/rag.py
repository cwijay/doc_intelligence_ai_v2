"""RAG (Retrieval-Augmented Generation) API schemas.

Multi-tenancy Architecture:
- One store per organization (created automatically)
- Documents organized in folders with metadata filtering
- Supports cross-folder and folder-specific searches
"""

from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field


# =============================================================================
# Request Models
# =============================================================================

class CreateStoreRequest(BaseModel):
    """Request to create a file search store.

    Note: In multi-tenant mode, one store is created per organization.
    """
    display_name: str = Field(..., description="Display name for the store", example="Company Policies")
    description: Optional[str] = Field(default=None, example="Store for company policy documents")


class UploadToStoreRequest(BaseModel):
    """Request to upload files to a store with folder organization.

    When using store_id='auto', the org_name field is required to create/reuse
    an org-specific Gemini File Search store named '<org_name>_file_search_store'.
    """
    file_paths: List[str] = Field(..., description="List of file paths to upload", example=["parsed/policy1.md", "parsed/policy2.md"])
    folder_id: Optional[str] = Field(None, description="Folder ID to organize documents", example="folder_123")
    folder_name: Optional[str] = Field(None, description="Folder name for metadata filtering", example="Invoices 2024")
    chunk_size: Optional[int] = Field(None, description="Chunk size for splitting", example=1000)
    chunk_overlap: Optional[int] = Field(None, description="Overlap between chunks", example=200)

    # Organization name for auto-store creation
    org_name: Optional[str] = Field(None, description="Organization name (required when store_id='auto')", example="ACME Corp")

    # Enhanced metadata for document traceability
    original_gcs_paths: Optional[List[str]] = Field(
        None,
        description="Original document GCS paths (before parsing), aligned with file_paths",
        example=["gs://bucket/org/original/doc.pdf"]
    )
    parser_version: Optional[str] = Field(
        default="llama_parse_v2.5",
        description="Parser version used for document parsing"
    )


class SearchStoreRequest(BaseModel):
    """Request to search a store with optional folder/file filtering.

    Search Scopes:
    - Single file: Set file_filter to search within a specific file
    - Folder: Set folder_name to search all files in a folder
    - Org-wide: Leave filters empty to search ALL indexed documents

    Search Modes:
    - semantic: Vector similarity search (default)
    - keyword: Keyword/BM25 search
    - hybrid: Combined semantic + keyword search
    """
    query: str = Field(..., description="Search query", example="What is the vacation policy?")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results", example=5)
    file_filter: Optional[str] = Field(None, description="Filter by filename", example="policy")
    folder_name: Optional[str] = Field(None, description="Filter by folder name (for folder-scoped search)", example="Invoices 2024")
    folder_id: Optional[str] = Field(None, description="Filter by folder ID", example="folder_123")
    search_mode: str = Field(
        default="semantic",
        description="Search mode: 'semantic' (vector), 'keyword' (BM25), or 'hybrid' (combined)",
        example="hybrid"
    )
    generate_answer: bool = Field(
        default=True,
        description="Whether to generate an answer from retrieved chunks",
        example=True
    )


# =============================================================================
# Response Models
# =============================================================================

class StoreInfo(BaseModel):
    """Information about a file search store."""
    store_id: str = Field(..., example="store_abc123")
    organization_id: Optional[str] = Field(default=None, description="Organization ID for multi-tenancy")
    gemini_store_id: Optional[str] = Field(default=None, description="Gemini File Search store ID")
    display_name: str = Field(..., example="Company Policies")
    description: Optional[str] = Field(default=None, example="Store for company policy documents")
    created_at: datetime
    updated_at: Optional[datetime] = None
    active_documents_count: int = Field(default=0, example=5)
    total_size_bytes: int = Field(default=0, example=1048576)
    status: str = Field(default="active", example="active")


class CreateStoreResponse(BaseModel):
    """Response for store creation."""
    success: bool = Field(..., example=True)
    store_id: Optional[str] = Field(default=None, example="store_abc123")
    display_name: Optional[str] = Field(default=None, example="Company Policies")
    created_at: Optional[datetime] = None
    error: Optional[str] = None


class ListStoresResponse(BaseModel):
    """Response for listing stores."""
    success: bool = Field(..., example=True)
    stores: List[StoreInfo] = Field(default_factory=list)
    count: int = Field(default=0, example=3)
    error: Optional[str] = None


class StoreFileInfo(BaseModel):
    """Information about a file in a store with enhanced metadata."""
    file_name: str = Field(..., example="policy1.md")
    file_size_bytes: Optional[int] = Field(default=None, example=15234)
    upload_date: Optional[datetime] = None
    chunk_count: Optional[int] = Field(default=None, example=12)
    status: str = Field(default="ready", example="ready")

    # Enhanced metadata for document traceability
    content_hash: Optional[str] = Field(None, description="SHA-256 content hash", example="a1b2c3...")
    original_file_extension: Optional[str] = Field(None, description="Original file extension", example=".pdf")
    original_file_size: Optional[int] = Field(None, description="Original file size in bytes", example=102400)
    original_gcs_path: Optional[str] = Field(None, description="Original document GCS path", example="gs://bucket/org/original/doc.pdf")
    parsed_gcs_path: Optional[str] = Field(None, description="Parsed document GCS path", example="gs://bucket/org/parsed/doc.md")
    parse_date: Optional[datetime] = Field(None, description="When document was parsed")
    parser_version: Optional[str] = Field(None, description="Parser version used", example="llama_parse_v2.5")
    org_name: Optional[str] = Field(None, description="Organization name", example="ACME Corp")
    folder_name: Optional[str] = Field(None, description="Folder name", example="Invoices 2024")


class UploadToStoreResponse(BaseModel):
    """Response for uploading files to a store."""
    success: bool = Field(..., example=True)
    store_id: str = Field(..., example="store_abc123")
    uploaded: int = Field(default=0, example=2)
    failed: int = Field(default=0, example=0)
    files: List[StoreFileInfo] = Field(default_factory=list)
    errors: List[Dict[str, str]] = Field(default_factory=list)
    error: Optional[str] = Field(None, description="Error message if upload failed")


class Citation(BaseModel):
    """A citation from search results."""
    file: str = Field(..., example="policy1.md")
    text: str = Field(..., example="Employees are entitled to 15 days of paid vacation per year...")
    relevance_score: Optional[float] = Field(default=None, example=0.92)
    page: Optional[int] = Field(default=None, example=3)
    chunk_id: Optional[str] = Field(default=None, example="chunk_xyz789")
    folder_name: Optional[str] = Field(default=None, description="Folder name for context", example="Invoices 2024")


class SearchStoreResponse(BaseModel):
    """Response for store search."""
    success: bool = Field(..., example=True)
    query: str = Field(..., example="What is the vacation policy?")
    response: Optional[str] = Field(default=None, example="According to the company policy, employees are entitled to 15 days...")
    citations: List[Citation] = Field(default_factory=list)
    processing_time_ms: float = Field(..., example=234.56)
    search_mode: str = Field(default="semantic", description="Search mode used", example="hybrid")
    error: Optional[str] = None


class ListStoreFilesResponse(BaseModel):
    """Response for listing files in a store."""
    success: bool = Field(..., example=True)
    store_id: str = Field(..., example="store_abc123")
    files: List[StoreFileInfo] = Field(default_factory=list)
    count: int = Field(default=0, example=5)
    error: Optional[str] = None


class DeleteStoreResponse(BaseModel):
    """Response for store deletion."""
    success: bool = Field(..., example=True)
    store_id: str = Field(..., example="store_abc123")
    message: Optional[str] = Field(default=None, example="Store deleted successfully")
    error: Optional[str] = None


# =============================================================================
# Folder Models
# =============================================================================

class CreateFolderRequest(BaseModel):
    """Request to create a document folder."""
    folder_name: str = Field(..., description="Name of the folder", example="Invoices 2024")
    description: Optional[str] = Field(None, description="Folder description", example="Invoices from fiscal year 2024")
    parent_folder_id: Optional[str] = Field(None, description="Parent folder ID for nested structure", example="folder_parent_123")


class UpdateFolderRequest(BaseModel):
    """Request to update a folder."""
    folder_name: Optional[str] = Field(None, description="New folder name", example="Invoices 2024 Q4")
    description: Optional[str] = Field(None, description="Updated description")


class FolderInfo(BaseModel):
    """Information about a document folder."""
    id: str = Field(..., example="folder_abc123", description="Folder ID")
    folder_id: str = Field(..., example="folder_abc123", description="Folder ID (alias for id)")
    organization_id: Optional[str] = Field(default=None, description="Organization ID for multi-tenancy")
    store_id: Optional[str] = Field(default=None, description="Associated file search store ID")
    folder_name: str = Field(..., example="Invoices 2024")
    description: Optional[str] = Field(None, example="Invoices from fiscal year 2024")
    parent_folder_id: Optional[str] = Field(None, example="folder_parent_123")
    document_count: int = Field(default=0, example=15)
    total_size_bytes: int = Field(default=0, example=1048576)
    created_at: datetime
    updated_at: datetime


class CreateFolderResponse(BaseModel):
    """Response for folder creation."""
    success: bool = Field(..., example=True)
    folder: Optional[FolderInfo] = None
    error: Optional[str] = None


class ListFoldersResponse(BaseModel):
    """Response for listing folders."""
    success: bool = Field(..., example=True)
    folders: List[FolderInfo] = Field(default_factory=list)
    count: int = Field(default=0, example=5)
    error: Optional[str] = None


class GetFolderResponse(BaseModel):
    """Response for getting a single folder."""
    success: bool = Field(..., example=True)
    folder: Optional[FolderInfo] = None
    error: Optional[str] = None


class DeleteFolderResponse(BaseModel):
    """Response for folder deletion."""
    success: bool = Field(..., example=True)
    folder_id: str = Field(..., example="folder_abc123")
    message: Optional[str] = Field(None, example="Folder deleted successfully")
    documents_deleted: int = Field(default=0, example=15)
    error: Optional[str] = None
