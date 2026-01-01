"""
API schemas for bulk processing endpoints.

Request and response models for the /api/v1/bulk/ endpoints.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field

from src.bulk.schemas import (
    BulkJobStatus,
    DocumentItemStatus,
    ProcessingOptions,
    BulkFolderInfo,
    SignedUrlInfo,
    BulkJobInfo,
    DocumentItemInfo,
)


# =============================================================================
# FOLDER MANAGEMENT
# =============================================================================


class CreateFolderRequest(BaseModel):
    """Request to create a bulk processing folder."""

    folder_name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Name for the bulk folder",
        examples=["invoices-2024", "contracts-q1"],
    )


class CreateFolderResponse(BaseModel):
    """Response after creating a bulk folder."""

    success: bool
    folder: BulkFolderInfo
    message: Optional[str] = None


class ListFoldersResponse(BaseModel):
    """Response listing bulk folders."""

    success: bool
    folders: List[BulkFolderInfo]
    count: int


class GenerateUploadUrlsRequest(BaseModel):
    """Request to generate signed URLs for upload."""

    filenames: List[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of filenames to generate upload URLs for",
    )
    expiration_minutes: Optional[int] = Field(
        default=60,
        ge=5,
        le=1440,
        description="URL expiration time in minutes (5-1440)",
    )


class GenerateUploadUrlsResponse(BaseModel):
    """Response with signed upload URLs."""

    success: bool
    folder_name: str
    urls: List[SignedUrlInfo]
    count: int
    message: Optional[str] = None


class ListFolderDocumentsResponse(BaseModel):
    """Response listing documents in a folder."""

    success: bool
    folder_name: str
    documents: List[str]
    count: int


# =============================================================================
# DIRECT UPLOAD
# =============================================================================


class UploadedFileInfo(BaseModel):
    """Info about a successfully uploaded file."""

    filename: str = Field(description="Sanitized filename")
    original_filename: str = Field(description="Original filename from upload")
    size_bytes: int = Field(description="File size in bytes")
    gcs_path: str = Field(description="GCS URI where file was saved")
    document_id: str = Field(description="Document item ID in database")


class FailedFileInfo(BaseModel):
    """Info about a file that failed to upload."""

    filename: str = Field(description="Filename that failed")
    error: str = Field(description="Error message describing the failure")


class BulkUploadResponse(BaseModel):
    """Response after bulk file upload."""

    success: bool = Field(description="True if at least one file uploaded successfully")
    job_id: Optional[str] = Field(default=None, description="Bulk job ID for tracking")
    folder_name: str = Field(description="Target folder name")
    total_documents: int = Field(default=0, description="Number of files successfully uploaded")
    uploaded_files: List[UploadedFileInfo] = Field(
        default_factory=list,
        description="Details of successfully uploaded files",
    )
    failed_files: List[FailedFileInfo] = Field(
        default_factory=list,
        description="Details of files that failed to upload",
    )
    status: Optional[BulkJobStatus] = Field(
        default=None,
        description="Job status (if auto_start=true)",
    )
    message: str = Field(description="Human-readable status message")


# =============================================================================
# JOB MANAGEMENT
# =============================================================================


class SubmitBulkJobRequest(BaseModel):
    """Request to submit a bulk processing job."""

    folder_name: str = Field(
        ...,
        description="Name of the bulk folder to process",
    )
    generate_summary: bool = Field(
        default=True,
        description="Generate summaries for documents",
    )
    generate_faqs: bool = Field(
        default=True,
        description="Generate FAQs for documents",
    )
    generate_questions: bool = Field(
        default=True,
        description="Generate comprehension questions",
    )
    num_faqs: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of FAQs to generate per document",
    )
    num_questions: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of questions to generate per document",
    )
    summary_max_words: int = Field(
        default=500,
        ge=50,
        le=2000,
        description="Maximum words in summary",
    )


class SubmitBulkJobResponse(BaseModel):
    """Response after submitting a bulk job."""

    success: bool
    job_id: str
    folder_name: str
    total_documents: int
    status: BulkJobStatus
    message: str


class BulkJobStatusResponse(BaseModel):
    """Response with bulk job status."""

    success: bool
    job: BulkJobInfo
    documents: Optional[List[DocumentItemInfo]] = Field(
        default=None,
        description="Document details (if include_documents=true)",
    )
    progress_percentage: float = Field(
        description="Processing progress percentage",
    )
    estimated_remaining_seconds: Optional[int] = Field(
        default=None,
        description="Estimated time remaining",
    )


class ListBulkJobsResponse(BaseModel):
    """Response listing bulk jobs."""

    success: bool
    jobs: List[BulkJobInfo]
    total: int
    limit: int
    offset: int


class CancelJobResponse(BaseModel):
    """Response after cancelling a job."""

    success: bool
    job_id: str
    message: str


class RetryDocumentRequest(BaseModel):
    """Request to retry failed documents."""

    document_ids: Optional[List[str]] = Field(
        default=None,
        description="Specific document IDs to retry (all failed if empty)",
    )


class RetryDocumentResponse(BaseModel):
    """Response after retrying documents."""

    success: bool
    retried_count: int
    message: str


# =============================================================================
# WEBHOOK
# =============================================================================


class WebhookDocumentUploadedRequest(BaseModel):
    """Request from Cloud Function when document is uploaded."""

    bucket: str = Field(description="GCS bucket name")
    name: str = Field(description="Object path in bucket")
    size: int = Field(default=0, description="File size in bytes")
    content_type: Optional[str] = Field(default=None)
    time_created: Optional[str] = Field(default=None)
    metageneration: Optional[str] = Field(default=None)


class WebhookResponse(BaseModel):
    """Response to webhook request."""

    success: bool
    message: str
    job_id: Optional[str] = None
    document_id: Optional[str] = None
