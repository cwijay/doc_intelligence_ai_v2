"""API schemas for content router endpoints."""

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class LoadParsedRequest(BaseModel):
    """Request to load pre-parsed content from GCS."""

    org_name: str = Field(
        ...,
        description="Organization name for path construction",
        example="Acme corp"
    )
    folder_name: str = Field(
        ...,
        description="Folder name where document is stored",
        example="invoices"
    )
    document_name: str = Field(
        ...,
        description="Original document name (with extension)",
        example="invoice_001.pdf"
    )

    @field_validator('org_name', 'folder_name', 'document_name')
    @classmethod
    def validate_path_components(cls, v: str) -> str:
        """Prevent path traversal attacks."""
        if '..' in v or v.startswith('/') or '\\' in v:
            raise ValueError("Invalid path component: path traversal not allowed")
        return v


class LoadParsedResponse(BaseModel):
    """Response with loaded parsed content - matches DocumentParseResponse format."""

    success: bool = Field(..., example=True)
    storage_path: str = Field(
        ...,
        description="Original file storage path",
        example="Acme corp/original/invoices/invoice_001.pdf"
    )
    parsed_storage_path: str = Field(
        ...,
        description="Parsed content storage path",
        example="Acme corp/parsed/invoices/invoice_001.md"
    )
    parsed_content: str = Field(
        ...,
        description="The markdown content from GCS"
    )
    parsing_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parsing metadata"
    )
    gcs_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="GCS file metadata"
    )
    file_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="File information"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Load timestamp"
    )
    error: Optional[str] = Field(default=None)


class CheckParsedExistsRequest(BaseModel):
    """Request to check if parsed content exists."""

    org_name: str = Field(..., description="Organization name")
    folder_name: str = Field(..., description="Folder name")
    document_name: str = Field(..., description="Document name")

    @field_validator('org_name', 'folder_name', 'document_name')
    @classmethod
    def validate_path_components(cls, v: str) -> str:
        """Prevent path traversal attacks."""
        if '..' in v or v.startswith('/') or '\\' in v:
            raise ValueError("Invalid path component: path traversal not allowed")
        return v


class CheckParsedExistsResponse(BaseModel):
    """Response for parsed content existence check."""

    exists: bool = Field(..., description="Whether parsed content exists")
    path: str = Field(..., description="The GCS path checked")
    error: Optional[str] = Field(default=None, description="Error message if check failed")


# Error responses for OpenAPI documentation
CONTENT_ERROR_RESPONSES = {
    400: {"description": "Invalid request parameters"},
    401: {"description": "Authentication required"},
    404: {"description": "Parsed content not found in GCS"},
    500: {"description": "Internal server error"},
}
