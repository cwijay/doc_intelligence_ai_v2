"""
Cloud Function: Bulk Document Upload Trigger

Triggered by GCS finalize events when documents are uploaded to bulk folders.
Sends webhook notification to the main application for processing.

Deployment:
    gcloud functions deploy bulk-document-trigger \
        --gen2 \
        --runtime=python312 \
        --region=us-central1 \
        --source=. \
        --entry-point=gcs_bulk_trigger \
        --trigger-event-filters="type=google.cloud.storage.object.v1.finalized" \
        --trigger-event-filters="bucket=biz2bricks-dev-v1-document-store" \
        --set-env-vars="WEBHOOK_URL=https://your-app-url/api/v1/bulk/webhook/document-uploaded,WEBHOOK_SECRET=your-secret"

Environment Variables:
    WEBHOOK_URL: URL of the bulk upload webhook endpoint
    WEBHOOK_SECRET: Shared secret for webhook authentication
    BULK_PATH_PREFIX: Path prefix for bulk folders (default: "bulk")
"""

import os
import logging
import json
from typing import Optional

import functions_framework
from cloudevents.http import CloudEvent
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from environment
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")
BULK_PATH_PREFIX = os.environ.get("BULK_PATH_PREFIX", "bulk")
WEBHOOK_TIMEOUT = int(os.environ.get("WEBHOOK_TIMEOUT", "30"))


def is_bulk_path(file_path: str) -> bool:
    """Check if file path is within a bulk folder."""
    parts = file_path.split("/")
    return len(parts) >= 3 and parts[1] == BULK_PATH_PREFIX


def extract_org_and_folder(file_path: str) -> tuple[Optional[str], Optional[str]]:
    """
    Extract org_id and folder_name from file path.

    Expected format: <org_id>/bulk/<folder_name>/<filename>

    Returns:
        Tuple of (org_id, folder_name) or (None, None) if invalid
    """
    parts = file_path.split("/")
    if len(parts) >= 4 and parts[1] == BULK_PATH_PREFIX:
        return parts[0], parts[2]
    return None, None


@functions_framework.cloud_event
def gcs_bulk_trigger(cloud_event: CloudEvent) -> None:
    """
    Cloud Function entry point for GCS finalize events.

    Triggered when a file upload is completed in the monitored bucket.
    Filters for bulk folder paths and sends webhook notification.

    Args:
        cloud_event: CloudEvent containing GCS object metadata
    """
    # Extract event data
    data = cloud_event.data
    bucket = data.get("bucket", "")
    file_path = data.get("name", "")
    file_size = int(data.get("size", 0))
    content_type = data.get("contentType", "")
    time_created = data.get("timeCreated", "")
    metageneration = data.get("metageneration", "")

    logger.info(f"GCS event received: bucket={bucket}, path={file_path}, size={file_size}")

    # Skip non-bulk paths
    if not is_bulk_path(file_path):
        logger.debug(f"Skipping non-bulk path: {file_path}")
        return

    # Skip directory markers (empty files ending with /)
    if file_path.endswith("/"):
        logger.debug(f"Skipping directory marker: {file_path}")
        return

    # Skip hidden files
    filename = file_path.split("/")[-1]
    if filename.startswith("."):
        logger.debug(f"Skipping hidden file: {file_path}")
        return

    # Validate we have webhook configuration
    if not WEBHOOK_URL:
        logger.error("WEBHOOK_URL environment variable not set")
        return

    # Extract org and folder for logging
    org_id, folder_name = extract_org_and_folder(file_path)
    logger.info(
        f"Processing bulk upload: org={org_id}, folder={folder_name}, "
        f"file={filename}, size={file_size}"
    )

    # Prepare webhook payload
    payload = {
        "bucket": bucket,
        "name": file_path,
        "size": file_size,
        "content_type": content_type,
        "time_created": time_created,
        "metageneration": metageneration,
    }

    # Prepare headers
    headers = {
        "Content-Type": "application/json",
    }
    if WEBHOOK_SECRET:
        headers["X-Webhook-Secret"] = WEBHOOK_SECRET

    # Send webhook notification
    try:
        response = requests.post(
            WEBHOOK_URL,
            json=payload,
            headers=headers,
            timeout=WEBHOOK_TIMEOUT,
        )

        if response.status_code == 200:
            result = response.json()
            logger.info(
                f"Webhook success: action={result.get('action', 'unknown')}, "
                f"job_id={result.get('job_id', 'N/A')}"
            )
        elif response.status_code == 401:
            logger.error("Webhook authentication failed - check WEBHOOK_SECRET")
        elif response.status_code == 403:
            logger.warning("Webhook disabled or forbidden")
        else:
            logger.error(
                f"Webhook failed: status={response.status_code}, "
                f"body={response.text[:500]}"
            )

    except requests.Timeout:
        logger.error(f"Webhook timeout after {WEBHOOK_TIMEOUT}s")
    except requests.RequestException as e:
        logger.error(f"Webhook request failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error sending webhook: {e}")


# HTTP trigger for testing
@functions_framework.http
def test_trigger(request):
    """
    HTTP endpoint for testing the webhook notification.

    POST /test with JSON body:
    {
        "bucket": "bucket-name",
        "name": "org123/bulk/folder/file.pdf",
        "size": 12345
    }
    """
    if request.method != "POST":
        return {"error": "POST required"}, 405

    try:
        data = request.get_json()
        if not data:
            return {"error": "JSON body required"}, 400

        bucket = data.get("bucket", "test-bucket")
        file_path = data.get("name", "")
        file_size = int(data.get("size", 0))

        if not file_path:
            return {"error": "name (file path) required"}, 400

        if not is_bulk_path(file_path):
            return {"error": "Not a bulk path", "path": file_path}, 400

        # Prepare payload
        payload = {
            "bucket": bucket,
            "name": file_path,
            "size": file_size,
            "content_type": data.get("content_type", "application/octet-stream"),
            "time_created": data.get("time_created", ""),
            "metageneration": data.get("metageneration", "1"),
        }

        # Send webhook
        headers = {"Content-Type": "application/json"}
        if WEBHOOK_SECRET:
            headers["X-Webhook-Secret"] = WEBHOOK_SECRET

        response = requests.post(
            WEBHOOK_URL,
            json=payload,
            headers=headers,
            timeout=WEBHOOK_TIMEOUT,
        )

        return {
            "success": response.status_code == 200,
            "status_code": response.status_code,
            "response": response.json() if response.status_code == 200 else response.text[:500],
        }

    except Exception as e:
        return {"error": str(e)}, 500
