"""
Webhook handler for Cloud Function notifications.

Handles GCS upload events from Cloud Functions and coordinates
bulk job creation and document tracking.
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple

from src.db.repositories import bulk_repository
from src.utils.gcs_utils import build_gcs_uri

from .config import get_bulk_config
from .schemas import BulkJobStatus, DocumentItemStatus, ProcessingOptions, BulkJobEvent

logger = logging.getLogger(__name__)

# Pattern to extract org_id and folder_name from GCS path
# Expected path format: <org_id>/bulk/<folder_name>/<filename>
BULK_PATH_PATTERN = re.compile(r"^([^/]+)/bulk/([^/]+)/([^/]+)$")

# Allowed file extensions for bulk processing
ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".pptx", ".ppt",
    ".xlsx", ".xls", ".txt", ".md", ".csv",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff",
}


async def handle_document_uploaded(
    bucket: str,
    file_path: str,
    file_size: int = 0,
) -> Dict[str, Any]:
    """
    Handle GCS upload notification from Cloud Function.

    This is called when a file is uploaded to a bulk folder in GCS.
    It creates or updates the bulk job and adds the document for processing.

    Args:
        bucket: GCS bucket name
        file_path: Object path in bucket (e.g., "org123/bulk/invoices/doc.pdf")
        file_size: File size in bytes

    Returns:
        Dict with job_id, document_id, message, and action taken

    Raises:
        ValueError: If path is not in expected bulk folder format
    """
    config = get_bulk_config()

    # Parse and validate the path
    org_id, folder_name, filename = _parse_bulk_path(file_path)

    # Validate file extension
    if not _is_allowed_file(filename):
        logger.info(f"Skipping unsupported file type: {filename}")
        return {
            "success": True,
            "message": f"File type not supported for bulk processing: {filename}",
            "action": "skipped",
        }

    # Build full GCS path
    gcs_uri = build_gcs_uri(bucket, file_path)

    logger.info(
        f"Processing uploaded document: org={org_id}, folder={folder_name}, "
        f"file={filename}, size={file_size}"
    )

    # Find existing pending/processing job for this folder
    job = await _find_or_create_job(org_id, folder_name, bucket)

    if not job:
        return {
            "success": False,
            "message": "Failed to create or find bulk job",
            "action": "error",
        }

    job_id = job["id"]

    # Check if document already exists in job
    existing_doc = await bulk_repository.get_document_item_by_path(job_id, gcs_uri)
    if existing_doc:
        logger.info(f"Document already exists in job: {gcs_uri}")
        return {
            "success": True,
            "job_id": job_id,
            "document_id": existing_doc["id"],
            "message": "Document already registered",
            "action": "exists",
        }

    # Check document count limit
    current_count = await bulk_repository.count_documents_in_job(job_id)
    if current_count >= config.max_documents_per_folder:
        logger.warning(
            f"Folder {folder_name} has reached document limit ({config.max_documents_per_folder})"
        )
        return {
            "success": False,
            "job_id": job_id,
            "message": f"Folder has reached maximum document limit ({config.max_documents_per_folder})",
            "action": "limit_exceeded",
        }

    # Add document to job
    doc_dict = await bulk_repository.create_document_item(
        bulk_job_id=job_id,
        original_path=gcs_uri,
        original_filename=filename,
    )

    # Update job total count
    await bulk_repository.increment_total_documents(job_id)

    logger.info(f"Added document {doc_dict['id']} to job {job_id}")

    # Check if we should auto-start the job
    action = "added"
    if config.auto_start_enabled and job["status"] == BulkJobStatus.PENDING.value:
        should_start = await _should_auto_start_job(job_id, config)
        if should_start:
            await _trigger_job_start(job_id)
            action = "added_and_started"
            logger.info(f"Auto-started job {job_id}")

    return {
        "success": True,
        "job_id": job_id,
        "document_id": doc_dict["id"],
        "message": f"Document added to job",
        "action": action,
    }


def _parse_bulk_path(file_path: str) -> Tuple[str, str, str]:
    """
    Parse GCS path to extract org_id, folder_name, and filename.

    Expected format: <org_id>/bulk/<folder_name>/<filename>

    Args:
        file_path: Object path in bucket

    Returns:
        Tuple of (org_id, folder_name, filename)

    Raises:
        ValueError: If path doesn't match expected format
    """
    match = BULK_PATH_PATTERN.match(file_path)
    if not match:
        raise ValueError(
            f"Invalid bulk folder path: {file_path}. "
            f"Expected format: <org_id>/bulk/<folder_name>/<filename>"
        )

    org_id, folder_name, filename = match.groups()
    return org_id, folder_name, filename


def _is_allowed_file(filename: str) -> bool:
    """Check if file extension is allowed for processing."""
    if not filename:
        return False
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in ALLOWED_EXTENSIONS


async def _find_or_create_job(
    org_id: str,
    folder_name: str,
    bucket: str,
) -> Optional[Dict[str, Any]]:
    """
    Find an existing pending/processing job for the folder or create a new one.

    Args:
        org_id: Organization ID
        folder_name: Bulk folder name
        bucket: GCS bucket name

    Returns:
        Job dict or None if creation failed
    """
    config = get_bulk_config()

    # Look for existing pending or processing job
    existing_job = await bulk_repository.find_active_job_for_folder(org_id, folder_name)

    if existing_job:
        logger.debug(f"Found existing job {existing_job['id']} for folder {folder_name}")
        return existing_job

    # Create new job
    source_path = build_gcs_uri(bucket, f"{org_id}/bulk/{folder_name}")

    # Use default processing options
    options = ProcessingOptions(
        generate_summary=config.default_generate_summary,
        generate_faqs=config.default_generate_faqs,
        generate_questions=config.default_generate_questions,
        num_faqs=config.default_num_faqs,
        num_questions=config.default_num_questions,
        summary_max_words=config.default_summary_max_words,
    )

    try:
        job_dict = await bulk_repository.create_bulk_job(
            organization_id=org_id,
            folder_name=folder_name,
            source_path=source_path,
            total_documents=0,  # Will be updated as documents are added
            options=options.model_dump(),
        )
        logger.info(f"Created new bulk job {job_dict['id']} for folder {folder_name}")
        return job_dict
    except Exception as e:
        logger.error(f"Failed to create bulk job: {e}")
        return None


async def _should_auto_start_job(job_id: str, config) -> bool:
    """
    Determine if job should be auto-started based on configuration.

    Auto-start conditions:
    1. Minimum document count reached
    2. OR delay since last upload exceeded

    Args:
        job_id: Job ID to check
        config: Bulk processing configuration

    Returns:
        True if job should be started
    """
    job = await bulk_repository.get_bulk_job(job_id)
    if not job:
        return False

    # Check minimum document count
    doc_count = await bulk_repository.count_documents_in_job(job_id)
    if doc_count >= config.auto_start_min_documents:
        # If we have enough documents, check if delay has passed
        if config.auto_start_delay_seconds > 0:
            # Get most recent document upload time
            last_doc = await bulk_repository.get_latest_document_in_job(job_id)
            if last_doc:
                last_upload = last_doc.get("created_at")
                if last_upload:
                    # Check if enough time has passed since last upload
                    if isinstance(last_upload, str):
                        last_upload = datetime.fromisoformat(last_upload.replace("Z", "+00:00"))
                    delay_threshold = datetime.utcnow() - timedelta(seconds=config.auto_start_delay_seconds)
                    if last_upload < delay_threshold:
                        return True
        else:
            # No delay configured, start immediately
            return True

    return False


async def _trigger_job_start(job_id: str) -> None:
    """
    Trigger job processing via the bulk queue.

    Args:
        job_id: Job ID to start
    """
    from .queue import get_bulk_queue

    queue = get_bulk_queue()
    queue.enqueue(BulkJobEvent(
        job_id=job_id,
        action="start",
    ))


async def validate_webhook_request(
    bucket: str,
    file_path: str,
    expected_bucket: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Validate incoming webhook request.

    Args:
        bucket: GCS bucket from request
        file_path: File path from request
        expected_bucket: Expected bucket name (from config)

    Returns:
        Tuple of (is_valid, error_message)
    """
    config = get_bulk_config()

    # Validate bucket matches expected
    if expected_bucket and bucket != expected_bucket:
        return False, f"Unexpected bucket: {bucket}"

    # Validate path is in bulk folder
    if "/bulk/" not in file_path:
        return False, "Not a bulk folder path"

    # Validate path format
    try:
        _parse_bulk_path(file_path)
    except ValueError as e:
        return False, str(e)

    return True, ""
