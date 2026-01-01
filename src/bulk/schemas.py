"""
Bulk processing Pydantic schemas.

These are domain models for the bulk processing module,
separate from API request/response schemas.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class BulkJobStatus(str, Enum):
    """Status values for bulk processing jobs."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    PARTIAL_FAILURE = "partial_failure"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DocumentItemStatus(str, Enum):
    """Status values for individual documents within a bulk job."""

    PENDING = "pending"
    PARSING = "parsing"
    PARSED = "parsed"
    INDEXING = "indexing"
    INDEXED = "indexed"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ProcessingOptions(BaseModel):
    """Options for bulk document processing."""

    generate_summary: bool = Field(
        default=True,
        description="Whether to generate document summaries",
    )
    generate_faqs: bool = Field(
        default=True,
        description="Whether to generate FAQs",
    )
    generate_questions: bool = Field(
        default=True,
        description="Whether to generate comprehension questions",
    )
    num_faqs: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of FAQs to generate",
    )
    num_questions: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of questions to generate",
    )
    summary_max_words: int = Field(
        default=500,
        ge=50,
        le=2000,
        description="Maximum words in summary",
    )


class BulkFolderInfo(BaseModel):
    """Information about a bulk processing folder."""

    folder_name: str = Field(description="Name of the folder")
    gcs_path: str = Field(description="Full GCS path to the folder")
    document_count: int = Field(default=0, description="Number of documents in folder")
    total_size_bytes: int = Field(default=0, description="Total size of documents")
    created_at: Optional[datetime] = Field(default=None, description="When folder was created")
    org_id: str = Field(description="Organization ID")


class SignedUrlInfo(BaseModel):
    """Information about a signed URL for upload."""

    filename: str = Field(description="Filename to upload")
    signed_url: str = Field(description="Signed URL for upload")
    gcs_path: str = Field(description="Destination GCS path")
    expires_at: datetime = Field(description="When the signed URL expires")
    content_type: Optional[str] = Field(default=None, description="Expected content type")


class DocumentItemInfo(BaseModel):
    """Information about a document within a bulk job."""

    id: str = Field(description="Document item ID")
    bulk_job_id: str = Field(description="Parent bulk job ID")
    original_path: str = Field(description="Original GCS path")
    original_filename: str = Field(description="Original filename")
    parsed_path: Optional[str] = Field(default=None, description="Path to parsed content")
    status: DocumentItemStatus = Field(default=DocumentItemStatus.PENDING)
    error_message: Optional[str] = Field(default=None)
    retry_count: int = Field(default=0)
    parse_time_ms: Optional[int] = Field(default=None)
    index_time_ms: Optional[int] = Field(default=None)
    generation_time_ms: Optional[int] = Field(default=None)
    total_time_ms: Optional[int] = Field(default=None)
    token_usage: int = Field(default=0)
    llamaparse_pages: int = Field(default=0)
    content_hash: Optional[str] = Field(default=None)
    created_at: Optional[datetime] = Field(default=None)
    updated_at: Optional[datetime] = Field(default=None)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentItemInfo":
        """Create from repository dictionary."""
        return cls(
            id=data["id"],
            bulk_job_id=data["bulk_job_id"],
            original_path=data["original_path"],
            original_filename=data["original_filename"],
            parsed_path=data.get("parsed_path"),
            status=DocumentItemStatus(data.get("status", "pending")),
            error_message=data.get("error_message"),
            retry_count=data.get("retry_count", 0),
            parse_time_ms=data.get("parse_time_ms"),
            index_time_ms=data.get("index_time_ms"),
            generation_time_ms=data.get("generation_time_ms"),
            total_time_ms=data.get("total_time_ms"),
            token_usage=data.get("token_usage", 0),
            llamaparse_pages=data.get("llamaparse_pages", 0),
            content_hash=data.get("content_hash"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )


class BulkJobInfo(BaseModel):
    """Information about a bulk processing job."""

    id: str = Field(description="Job ID")
    organization_id: str = Field(description="Organization ID")
    folder_name: str = Field(description="Bulk folder name")
    source_path: str = Field(description="GCS source path")
    total_documents: int = Field(default=0)
    completed_count: int = Field(default=0)
    failed_count: int = Field(default=0)
    skipped_count: int = Field(default=0)
    status: BulkJobStatus = Field(default=BulkJobStatus.PENDING)
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    error_message: Optional[str] = Field(default=None)
    total_tokens_used: int = Field(default=0)
    total_llamaparse_pages: int = Field(default=0)
    options: ProcessingOptions = Field(default_factory=ProcessingOptions)
    created_at: Optional[datetime] = Field(default=None)
    updated_at: Optional[datetime] = Field(default=None)
    documents: Optional[List[DocumentItemInfo]] = Field(
        default=None,
        description="Document items (only included if requested)",
    )

    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage."""
        if self.total_documents == 0:
            return 0.0
        processed = self.completed_count + self.failed_count + self.skipped_count
        return round((processed / self.total_documents) * 100, 1)

    @property
    def pending_count(self) -> int:
        """Calculate pending document count."""
        processed = self.completed_count + self.failed_count + self.skipped_count
        return max(0, self.total_documents - processed)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BulkJobInfo":
        """Create from repository dictionary."""
        options_data = data.get("options", {})
        return cls(
            id=data["id"],
            organization_id=data["organization_id"],
            folder_name=data["folder_name"],
            source_path=data["source_path"],
            total_documents=data.get("total_documents", 0),
            completed_count=data.get("completed_count", 0),
            failed_count=data.get("failed_count", 0),
            skipped_count=data.get("skipped_count", 0),
            status=BulkJobStatus(data.get("status", "pending")),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            error_message=data.get("error_message"),
            total_tokens_used=data.get("total_tokens_used", 0),
            total_llamaparse_pages=data.get("total_llamaparse_pages", 0),
            options=ProcessingOptions(**options_data) if options_data else ProcessingOptions(),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )


class DocumentProcessingEvent(BaseModel):
    """Event for document processing queue."""

    job_id: str = Field(description="Bulk job ID")
    document_id: str = Field(description="Document item ID")
    action: str = Field(
        description="Action to perform: process, retry, skip",
        default="process",
    )
    retry_count: int = Field(default=0)


class BulkJobEvent(BaseModel):
    """Event for bulk job queue."""

    job_id: str = Field(description="Bulk job ID")
    action: str = Field(
        description="Action to perform: start, process_document, complete, cancel",
    )
    document_id: Optional[str] = Field(
        default=None,
        description="Document ID (for process_document action)",
    )


class WebhookPayload(BaseModel):
    """Payload from Cloud Function webhook."""

    bucket: str = Field(description="GCS bucket name")
    name: str = Field(description="Object name (file path)")
    size: int = Field(default=0, description="File size in bytes")
    content_type: Optional[str] = Field(default=None)
    time_created: Optional[str] = Field(default=None)
    updated: Optional[str] = Field(default=None)
    metageneration: Optional[str] = Field(default=None)

    @property
    def org_id(self) -> Optional[str]:
        """Extract organization ID from path."""
        # Path format: org_id/bulk/folder_name/filename
        parts = self.name.split("/")
        return parts[0] if len(parts) >= 3 else None

    @property
    def folder_name(self) -> Optional[str]:
        """Extract folder name from path."""
        # Path format: org_id/bulk/folder_name/filename
        parts = self.name.split("/")
        return parts[2] if len(parts) >= 4 else None

    @property
    def filename(self) -> Optional[str]:
        """Extract filename from path."""
        parts = self.name.split("/")
        return parts[-1] if parts else None
