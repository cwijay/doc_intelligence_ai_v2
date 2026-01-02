"""
Bulk processing API router.

Endpoints for bulk document processing:
- Direct upload (new, simplified approach)
- Folder management (create, list, upload URLs)
- Job management (submit, status, cancel, retry)
- Webhook for Cloud Function triggers (legacy)
"""

import logging
from pathlib import Path as FilePath
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query, Path, Header, UploadFile, File, Form

from ..dependencies import get_org_id
from ..schemas.bulk import (
    CreateFolderRequest,
    CreateFolderResponse,
    ListFoldersResponse,
    GenerateUploadUrlsRequest,
    GenerateUploadUrlsResponse,
    ListFolderDocumentsResponse,
    SubmitBulkJobRequest,
    SubmitBulkJobResponse,
    BulkJobStatusResponse,
    ListBulkJobsResponse,
    CancelJobResponse,
    RetryDocumentRequest,
    RetryDocumentResponse,
    WebhookDocumentUploadedRequest,
    WebhookResponse,
    # New upload schemas
    BulkUploadResponse,
    UploadedFileInfo,
    FailedFileInfo,
)
from src.bulk.schemas import ProcessingOptions, BulkJobStatus, BulkJobEvent
from src.bulk.folder_manager import get_folder_manager
from src.bulk.service import get_bulk_service
from src.bulk.queue import get_bulk_queue
from src.bulk.config import get_bulk_config
from src.storage import get_storage
from src.db.repositories import bulk_repository

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# DIRECT UPLOAD (Preferred method - no signed URLs required)
# =============================================================================


def _is_supported_file(filename: str, supported_extensions: List[str]) -> bool:
    """Check if file extension is supported."""
    if not filename:
        return False
    ext = FilePath(filename).suffix.lower()
    return ext in supported_extensions


def _sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    # Replace spaces with underscores, remove problematic characters
    safe_name = filename.replace(" ", "_")
    # Keep only alphanumeric, underscore, hyphen, and dot
    safe_name = "".join(
        c for c in safe_name if c.isalnum() or c in ("_", "-", ".")
    )
    return safe_name or "unnamed_file"


@router.post(
    "/upload",
    response_model=BulkUploadResponse,
    operation_id="uploadBulkFiles",
    summary="Upload files for bulk processing",
    tags=["Bulk Upload"],
)
async def upload_bulk_files(
    folder_name: str = Form(..., description="Target folder name for documents"),
    org_name: str = Form(..., description="Organization name for GCS path"),
    files: List[UploadFile] = File(..., description="Files to upload (max 10)"),
    generate_summary: bool = Form(True, description="Generate summaries"),
    generate_faqs: bool = Form(True, description="Generate FAQs"),
    generate_questions: bool = Form(True, description="Generate questions"),
    num_faqs: int = Form(10, ge=1, le=50, description="Number of FAQs per document"),
    num_questions: int = Form(10, ge=1, le=100, description="Number of questions per document"),
    summary_max_words: int = Form(500, ge=50, le=2000, description="Max words in summary"),
    auto_start: bool = Form(True, description="Start processing immediately"),
    org_id: str = Depends(get_org_id),
):
    """
    Upload multiple files and start bulk processing job.

    **Preferred Method**: This is the simplified direct upload approach.
    No signed URLs or Cloud Functions required.

    **Workflow**:
    1. Validates file count (max 10) and file types
    2. Saves files to GCS at `/<org>/original/<folder>/`
    3. Creates bulk job and document records in database
    4. Optionally starts processing immediately (auto_start=true)

    **GCS Path**: `gs://bucket/<org_name>/original/<folder_name>/<filename>`

    **Processing** (async, if auto_start=true):
    - Parses each document via LlamaParse
    - Indexes to Gemini File Search store
    - Generates summary, FAQs, questions (based on options)

    Poll `GET /jobs/{job_id}` for processing status.
    """
    config = get_bulk_config()
    storage = get_storage()

    # Validate file count
    if len(files) > config.max_documents_per_folder:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum is {config.max_documents_per_folder}",
        )

    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")

    # Validate folder name
    if not folder_name or not folder_name.strip():
        raise HTTPException(status_code=400, detail="Folder name is required")

    # Validate org_name
    if not org_name or not org_name.strip():
        raise HTTPException(status_code=400, detail="Organization name is required")

    folder_name = folder_name.strip()
    org_name = org_name.strip()

    # Create processing options
    options = ProcessingOptions(
        generate_summary=generate_summary,
        generate_faqs=generate_faqs,
        generate_questions=generate_questions,
        num_faqs=num_faqs,
        num_questions=num_questions,
        summary_max_words=summary_max_words,
    )

    # Create bulk job record (status=pending, will update count later)
    source_path = f"gs://{storage.bucket_name}/{org_name}/original/{folder_name}"

    job_dict = await bulk_repository.create_bulk_job(
        organization_id=org_id,
        folder_name=folder_name,
        source_path=source_path,
        total_documents=0,  # Will be updated after uploads
        options=options.model_dump(),
    )
    job_id = job_dict["id"]

    uploaded_files: List[UploadedFileInfo] = []
    failed_files: List[FailedFileInfo] = []

    # Process each file
    for file in files:
        try:
            original_filename = file.filename or "unnamed"

            # Validate file type
            if not _is_supported_file(original_filename, config.supported_extensions):
                ext = FilePath(original_filename).suffix.lower() if original_filename else "unknown"
                failed_files.append(FailedFileInfo(
                    filename=original_filename,
                    error=f"Unsupported file type: {ext}. Supported: {', '.join(config.supported_extensions[:5])}...",
                ))
                continue

            # Read file content
            content = await file.read()

            # Validate file size
            max_size_bytes = config.max_file_size_mb * 1024 * 1024
            if len(content) > max_size_bytes:
                failed_files.append(FailedFileInfo(
                    filename=original_filename,
                    error=f"File exceeds {config.max_file_size_mb}MB limit ({len(content) / 1024 / 1024:.1f}MB)",
                ))
                continue

            # Sanitize filename
            safe_filename = _sanitize_filename(original_filename)

            # Save to GCS: /<org>/original/<folder>/<filename>
            directory = f"{org_name}/original/{folder_name}"
            gcs_path = await storage.save_bytes(
                content=content,
                filename=safe_filename,
                directory=directory,
                use_prefix=False,
            )

            # Create document record
            doc_dict = await bulk_repository.create_document_item(
                bulk_job_id=job_id,
                original_path=gcs_path,
                original_filename=original_filename,
            )

            uploaded_files.append(UploadedFileInfo(
                filename=safe_filename,
                original_filename=original_filename,
                size_bytes=len(content),
                gcs_path=gcs_path,
                document_id=doc_dict["id"],
            ))

            logger.info(f"Uploaded {original_filename} to {gcs_path}")

        except Exception as e:
            logger.error(f"Failed to upload {file.filename}: {e}")
            failed_files.append(FailedFileInfo(
                filename=file.filename or "unknown",
                error=str(e),
            ))

    # Update job with actual document count
    await bulk_repository.update_job_document_count(
        job_id,
        total_documents=len(uploaded_files),
    )

    # Start processing if requested and we have uploaded files
    status = BulkJobStatus.PENDING
    if auto_start and len(uploaded_files) > 0:
        queue = get_bulk_queue()
        queue.enqueue(BulkJobEvent(job_id=job_id, action="start"))
        status = BulkJobStatus.PROCESSING
        logger.info(f"Enqueued bulk job {job_id} for processing ({len(uploaded_files)} files)")

    # Build response message
    if len(uploaded_files) > 0 and len(failed_files) == 0:
        message = f"Uploaded {len(uploaded_files)} files. {'Processing started.' if auto_start else 'Ready to process.'}"
    elif len(uploaded_files) > 0 and len(failed_files) > 0:
        message = f"Uploaded {len(uploaded_files)} files, {len(failed_files)} failed. {'Processing started.' if auto_start else 'Ready to process.'}"
    else:
        message = f"All {len(failed_files)} files failed to upload."

    return BulkUploadResponse(
        success=len(uploaded_files) > 0,
        job_id=job_id,
        folder_name=folder_name,
        total_documents=len(uploaded_files),
        uploaded_files=uploaded_files,
        failed_files=failed_files,
        status=status if len(uploaded_files) > 0 else None,
        message=message,
    )


# =============================================================================
# FOLDER MANAGEMENT
# =============================================================================


@router.post(
    "/folders",
    response_model=CreateFolderResponse,
    operation_id="createBulkFolder",
    summary="Create bulk processing folder",
    tags=["Bulk Folders"],
)
async def create_bulk_folder(
    request: CreateFolderRequest,
    org_id: str = Depends(get_org_id),
):
    """
    Create a new folder for bulk document processing.

    **Path Structure**: `gs://bucket/<org_id>/bulk/<folder_name>/`

    After creation, use the upload-urls endpoint to get signed URLs
    for direct document upload.
    """
    folder_manager = get_folder_manager()

    # Check if folder already exists
    exists = await folder_manager.folder_exists(org_id, request.folder_name)
    if exists:
        raise HTTPException(
            status_code=400,
            detail=f"Folder '{request.folder_name}' already exists",
        )

    folder_info = await folder_manager.create_folder(org_id, request.folder_name)

    return CreateFolderResponse(
        success=True,
        folder=folder_info,
        message=f"Folder '{request.folder_name}' created successfully",
    )


@router.get(
    "/folders",
    response_model=ListFoldersResponse,
    operation_id="listBulkFolders",
    summary="List bulk processing folders",
    tags=["Bulk Folders"],
)
async def list_bulk_folders(
    org_id: str = Depends(get_org_id),
):
    """List all bulk processing folders for the organization."""
    folder_manager = get_folder_manager()
    folders = await folder_manager.list_folders(org_id)

    return ListFoldersResponse(
        success=True,
        folders=folders,
        count=len(folders),
    )


@router.post(
    "/folders/{folder_name}/upload-urls",
    response_model=GenerateUploadUrlsResponse,
    operation_id="generateUploadUrls",
    summary="Generate signed URLs for upload",
    tags=["Bulk Folders"],
)
async def generate_upload_urls(
    folder_name: str = Path(..., description="Folder name"),
    request: GenerateUploadUrlsRequest = ...,
    org_id: str = Depends(get_org_id),
):
    """
    Generate signed URLs for direct GCS upload.

    Use these URLs to upload documents directly to GCS without
    going through our API. Each URL is valid for the specified
    expiration time (default: 60 minutes).

    **Upload Method**: PUT request to signed URL with file content.
    """
    folder_manager = get_folder_manager()
    config = get_bulk_config()

    # Validate document count limit
    if len(request.filenames) > config.max_documents_per_folder:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum is {config.max_documents_per_folder}",
        )

    # Generate signed URLs
    signed_urls = await folder_manager.generate_upload_urls(
        org_id=org_id,
        folder_name=folder_name,
        filenames=request.filenames,
        expiration_minutes=request.expiration_minutes,
    )

    if not signed_urls:
        raise HTTPException(
            status_code=400,
            detail="No valid files to upload. Check file extensions.",
        )

    return GenerateUploadUrlsResponse(
        success=True,
        folder_name=folder_name,
        urls=signed_urls,
        count=len(signed_urls),
    )


@router.get(
    "/folders/{folder_name}/documents",
    response_model=ListFolderDocumentsResponse,
    operation_id="listFolderDocuments",
    summary="List documents in bulk folder",
    tags=["Bulk Folders"],
)
async def list_folder_documents(
    folder_name: str = Path(..., description="Folder name"),
    org_id: str = Depends(get_org_id),
):
    """List all documents in a bulk processing folder."""
    folder_manager = get_folder_manager()
    documents = await folder_manager.list_documents(org_id, folder_name)

    return ListFolderDocumentsResponse(
        success=True,
        folder_name=folder_name,
        documents=documents,
        count=len(documents),
    )


# =============================================================================
# JOB MANAGEMENT
# =============================================================================


@router.post(
    "/jobs",
    response_model=SubmitBulkJobResponse,
    operation_id="submitBulkJob",
    summary="Submit bulk processing job",
    tags=["Bulk Jobs"],
)
async def submit_bulk_job(
    request: SubmitBulkJobRequest,
    org_id: str = Depends(get_org_id),
):
    """
    Submit a bulk document processing job.

    **Workflow**:
    1. Validates folder exists and meets document limit
    2. Checks quota (LlamaParse pages, tokens)
    3. Creates job record and queues for processing
    4. Returns job_id for status polling

    **Processing** (async):
    - Parses each document via LlamaParse
    - Saves parsed content to `/<org>/parsed/<folder>/<file>.md`
    - Indexes to Gemini File Search store
    - Generates summary, FAQs, questions (based on options)
    """
    service = get_bulk_service()

    options = ProcessingOptions(
        generate_summary=request.generate_summary,
        generate_faqs=request.generate_faqs,
        generate_questions=request.generate_questions,
        num_faqs=request.num_faqs,
        num_questions=request.num_questions,
        summary_max_words=request.summary_max_words,
    )

    try:
        job = await service.create_job(
            org_id=org_id,
            folder_name=request.folder_name,
            options=options,
        )

        # Enqueue job for processing
        queue = get_bulk_queue()
        queue.enqueue(BulkJobEvent(
            job_id=job.id,
            action="start",
        ))

        return SubmitBulkJobResponse(
            success=True,
            job_id=job.id,
            folder_name=job.folder_name,
            total_documents=job.total_documents,
            status=job.status,
            message=f"Job submitted. Processing {job.total_documents} documents.",
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to submit bulk job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/jobs/{job_id}",
    response_model=BulkJobStatusResponse,
    operation_id="getBulkJobStatus",
    summary="Get bulk job status",
    tags=["Bulk Jobs"],
)
async def get_bulk_job_status(
    job_id: str = Path(..., description="Job ID"),
    include_documents: bool = Query(
        False,
        description="Include document-level details",
    ),
    org_id: str = Depends(get_org_id),
):
    """
    Get status of a bulk processing job.

    **Progress Tracking**:
    - completed_count, failed_count, pending_count
    - Per-document status (if include_documents=true)
    - Progress percentage
    - Total token usage
    """
    service = get_bulk_service()

    job = await service.get_job_status(job_id, include_documents=include_documents)

    if not job or job.organization_id != org_id:
        raise HTTPException(status_code=404, detail="Job not found")

    # Estimate remaining time based on average processing time
    estimated_remaining = None
    if job.status == BulkJobStatus.PROCESSING and job.completed_count > 0:
        # Calculate average time per document
        # This is a rough estimate based on progress
        pass

    return BulkJobStatusResponse(
        success=True,
        job=job,
        documents=job.documents,
        progress_percentage=job.progress_percentage,
        estimated_remaining_seconds=estimated_remaining,
    )


@router.get(
    "/jobs",
    response_model=ListBulkJobsResponse,
    operation_id="listBulkJobs",
    summary="List bulk processing jobs",
    tags=["Bulk Jobs"],
)
async def list_bulk_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    org_id: str = Depends(get_org_id),
):
    """List bulk processing jobs for the organization."""
    from src.db.repositories import bulk_repository

    jobs_dict = await bulk_repository.list_bulk_jobs(
        organization_id=org_id,
        status=status,
        limit=limit,
        offset=offset,
    )

    total = await bulk_repository.count_bulk_jobs(org_id, status)

    from src.bulk.schemas import BulkJobInfo

    jobs = [BulkJobInfo.from_dict(j) for j in jobs_dict]

    return ListBulkJobsResponse(
        success=True,
        jobs=jobs,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post(
    "/jobs/{job_id}/cancel",
    response_model=CancelJobResponse,
    operation_id="cancelBulkJob",
    summary="Cancel bulk processing job",
    tags=["Bulk Jobs"],
)
async def cancel_bulk_job(
    job_id: str = Path(..., description="Job ID"),
    org_id: str = Depends(get_org_id),
):
    """Cancel a running bulk processing job."""
    from src.db.repositories import bulk_repository

    job_dict = await bulk_repository.get_bulk_job(job_id)

    if not job_dict or job_dict["organization_id"] != org_id:
        raise HTTPException(status_code=404, detail="Job not found")

    # Idempotent: if already cancelled, return success
    if job_dict["status"] == "cancelled":
        return CancelJobResponse(
            success=True,
            job_id=job_id,
            message="Job is already cancelled",
        )

    # Cannot cancel completed or failed jobs
    if job_dict["status"] not in ["pending", "processing"]:
        raise HTTPException(
            status_code=400,
            detail=f"Job cannot be cancelled (status: {job_dict['status']})",
        )

    queue = get_bulk_queue()
    queue.enqueue(BulkJobEvent(
        job_id=job_id,
        action="cancel",
    ))

    return CancelJobResponse(
        success=True,
        job_id=job_id,
        message="Job cancellation requested",
    )


@router.post(
    "/jobs/{job_id}/retry",
    response_model=RetryDocumentResponse,
    operation_id="retryFailedDocuments",
    summary="Retry failed documents in job",
    tags=["Bulk Jobs"],
)
async def retry_failed_documents(
    job_id: str = Path(..., description="Job ID"),
    request: Optional[RetryDocumentRequest] = None,
    org_id: str = Depends(get_org_id),
):
    """Retry failed documents in a bulk job."""
    from src.db.repositories import bulk_repository

    job_dict = await bulk_repository.get_bulk_job(job_id)

    if not job_dict or job_dict["organization_id"] != org_id:
        raise HTTPException(status_code=404, detail="Job not found")

    service = get_bulk_service()

    document_ids = request.document_ids if request else None
    count = await service.retry_failed_documents(job_id, document_ids)

    return RetryDocumentResponse(
        success=True,
        retried_count=count,
        message=f"Queued {count} documents for retry",
    )


# =============================================================================
# WEBHOOK
# =============================================================================


@router.post(
    "/webhook/document-uploaded",
    response_model=WebhookResponse,
    operation_id="webhookDocumentUploaded",
    summary="Webhook for document upload notification",
    tags=["Bulk Webhook"],
    include_in_schema=True,  # Set to False to hide from API docs
)
async def webhook_document_uploaded(
    request: WebhookDocumentUploadedRequest,
    x_webhook_secret: Optional[str] = Header(None, alias="X-Webhook-Secret"),
):
    """
    Webhook endpoint for Cloud Function to notify when a document is uploaded.

    Called automatically by the Cloud Function when a file is uploaded
    to the bulk folder in GCS.
    """
    config = get_bulk_config()

    if not config.webhook_enabled:
        raise HTTPException(status_code=403, detail="Webhook is disabled")

    # Verify webhook secret if configured
    if config.webhook_secret:
        if x_webhook_secret != config.webhook_secret:
            raise HTTPException(status_code=401, detail="Invalid webhook secret")

    # Import webhook handler
    from src.bulk.webhook_handler import handle_document_uploaded

    try:
        result = await handle_document_uploaded(
            bucket=request.bucket,
            file_path=request.name,
            file_size=request.size,
        )

        return WebhookResponse(
            success=True,
            message=result.get("message", "Document processed"),
            job_id=result.get("job_id"),
            document_id=result.get("document_id"),
        )

    except ValueError as e:
        logger.warning(f"Webhook validation error: {e}")
        return WebhookResponse(
            success=False,
            message=str(e),
        )
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
