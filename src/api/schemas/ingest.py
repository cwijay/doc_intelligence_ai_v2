"""Document Ingestion API schemas."""

from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field


# =============================================================================
# Request Models
# =============================================================================

class ParseRequest(BaseModel):
    """Request to parse a document."""
    file_path: str = Field(..., description="Path to the file to parse", example="upload/document.pdf")
    folder_name: str = Field(..., description="Folder name for organizing parsed documents", example="invoices")
    output_format: str = Field(default="markdown", description="Output format", example="markdown")
    language: str = Field(default="en", description="Document language", example="en")
    save_to_parsed: bool = Field(default=True, description="Save to parsed directory", example=True)


class ListFilesRequest(BaseModel):
    """Request to list files."""
    directory: str = Field(default="parsed", description="Directory to list (parsed or upload)", example="parsed")
    extension: Optional[str] = Field(None, description="Filter by extension (.md, .pdf, etc.)", example=".md")


# =============================================================================
# Response Models
# =============================================================================

class UploadedFile(BaseModel):
    """Information about an uploaded file."""
    filename: str = Field(..., example="doc_abc123.pdf")
    original_filename: str = Field(..., example="Annual Report 2024.pdf")
    size_bytes: int = Field(..., example=1548234)
    path: str = Field(..., example="upload/doc_abc123.pdf")
    content_type: Optional[str] = Field(default=None, example="application/pdf")
    uploaded_at: datetime


class UploadResponse(BaseModel):
    """Response for file upload."""
    success: bool = Field(..., example=True)
    files: List[UploadedFile] = Field(default_factory=list)
    failed: List[Dict[str, str]] = Field(default_factory=list)
    message: Optional[str] = Field(default=None, example="2 files uploaded successfully")


class ParseResponse(BaseModel):
    """Response for document parsing."""
    success: bool = Field(..., example=True)
    file_path: str = Field(..., example="upload/document.pdf")
    output_path: Optional[str] = Field(default=None, example="parsed/document.md")
    parsed_content: Optional[str] = None
    content_preview: Optional[str] = Field(None, description="First 500 chars", example="# Document Title\n\nThis document covers...")
    pages: Optional[int] = Field(default=None, example=12)
    format: str = Field(..., example="markdown")
    extraction_time_ms: float = Field(..., example=3456.78)
    error: Optional[str] = None


class FileInfo(BaseModel):
    """Information about a file."""
    name: str = Field(..., example="Sample1.md")
    path: str = Field(..., example="parsed/Sample1.md")
    size_bytes: int = Field(..., example=15234)
    extension: str = Field(..., example=".md")
    modified_at: datetime
    is_parsed: bool = Field(default=False, example=True)
    # Status tracking for upload/parse lifecycle
    status: str = Field(default="uploaded", description="Document status", example="parsed")
    parsed_path: Optional[str] = Field(default=None, description="Path to parsed .md file", example="gs://bucket/org/parsed/invoices/Sample1.md")
    parsed_at: Optional[datetime] = Field(default=None, description="Timestamp when parsing completed")


class ListFilesResponse(BaseModel):
    """Response for listing files."""
    success: bool = Field(..., example=True)
    directory: str = Field(..., example="parsed")
    files: List[FileInfo] = Field(default_factory=list)
    count: int = Field(default=0, example=5)
    error: Optional[str] = None


# =============================================================================
# Save and Index Models
# =============================================================================

class SaveAndIndexRequest(BaseModel):
    """Request to save parsed content to GCS and index in Gemini File Search store."""
    content: str = Field(..., description="Parsed/edited markdown content")
    target_path: str = Field(
        ...,
        description="Target path in GCS (e.g., 'Acme Corp/parsed/invoices/Sample.md')",
        example="Acme Corp/parsed/invoices/Sample.md"
    )
    org_name: str = Field(..., description="Organization name for store naming", example="Acme Corp")
    folder_name: str = Field(..., description="Folder name for GCS path and metadata", example="invoices")
    original_filename: str = Field(..., description="Original document filename", example="Sample.pdf")
    original_gcs_path: Optional[str] = Field(
        None,
        description="Original file GCS path",
        example="gs://bucket/Acme Corp/original/invoices/Sample.pdf"
    )
    parser_version: Optional[str] = Field(
        default="llama_parse_v2.5",
        description="Parser version for metadata",
        example="llama_parse_v2.5"
    )
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class SaveAndIndexResponse(BaseModel):
    """Response for save and index operation."""
    success: bool = Field(..., example=True)
    saved_path: Optional[str] = Field(
        None,
        description="Full GCS path where content was saved",
        example="gs://bucket/Acme Corp/parsed/invoices/Sample.md"
    )
    store_id: Optional[str] = Field(None, description="Gemini File Search store ID", example="abc123")
    store_name: Optional[str] = Field(
        None,
        description="Gemini File Search store name",
        example="Acme Corp_file_search_store"
    )
    indexed: bool = Field(default=False, description="Whether document was indexed in store")
    message: Optional[str] = Field(None, example="Document saved and indexed successfully")
    error: Optional[str] = Field(None, description="Error message if operation failed")
