"""
Bulk folder management with signed URL generation.

Handles GCS folder operations for bulk document processing:
- Create/list/validate bulk folders
- Generate signed URLs for direct upload
- List documents in folders
"""

import asyncio
import logging
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Optional, List

from google.cloud import storage
from google.cloud.exceptions import NotFound

from src.core.executors import get_executors
from src.storage import get_storage
from src.rag.llama_parse_util import SUPPORTED_EXTENSIONS

from .config import get_bulk_config, BulkProcessingConfig
from .schemas import BulkFolderInfo, SignedUrlInfo

logger = logging.getLogger(__name__)


class FolderManager:
    """Manages bulk processing folders in GCS."""

    def __init__(self, config: Optional[BulkProcessingConfig] = None):
        """
        Initialize folder manager.

        Args:
            config: Bulk processing config (uses singleton if not provided)
        """
        self.config = config or get_bulk_config()
        self._storage = get_storage()
        self._bucket_name = self._storage.bucket_name

        # Get GCS client for signed URL generation
        self._client = storage.Client()
        self._bucket = self._client.bucket(self._bucket_name)

        logger.info(f"FolderManager initialized for bucket: {self._bucket_name}")

    def _get_bulk_path(self, org_id: str, folder_name: str) -> str:
        """Construct the bulk folder path."""
        # Path format: org_id/bulk/folder_name/
        return f"{org_id}/{self.config.bulk_folder_prefix}/{folder_name}"

    def _get_full_gcs_uri(self, path: str) -> str:
        """Get full GCS URI from path."""
        return f"gs://{self._bucket_name}/{path}"

    async def create_folder(
        self,
        org_id: str,
        folder_name: str,
    ) -> BulkFolderInfo:
        """
        Create a bulk processing folder.

        Creates a placeholder file to establish the folder in GCS.

        Args:
            org_id: Organization ID
            folder_name: Name for the bulk folder

        Returns:
            Created folder info
        """
        folder_path = self._get_bulk_path(org_id, folder_name)
        placeholder_path = f"{folder_path}/.folder_created"

        # Create placeholder blob
        blob = self._bucket.blob(placeholder_path)
        content = f"Created: {datetime.utcnow().isoformat()}"

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            get_executors().io_executor,
            partial(blob.upload_from_string, content, content_type="text/plain"),
        )

        logger.info(f"Created bulk folder: {folder_path}")

        return BulkFolderInfo(
            folder_name=folder_name,
            gcs_path=self._get_full_gcs_uri(folder_path),
            document_count=0,
            total_size_bytes=0,
            created_at=datetime.utcnow(),
            org_id=org_id,
        )

    async def get_folder_info(
        self,
        org_id: str,
        folder_name: str,
    ) -> Optional[BulkFolderInfo]:
        """
        Get information about a bulk folder.

        Args:
            org_id: Organization ID
            folder_name: Name of the folder

        Returns:
            Folder info or None if not found
        """
        folder_path = self._get_bulk_path(org_id, folder_name)

        loop = asyncio.get_running_loop()

        # List blobs in folder
        def list_blobs():
            prefix = f"{folder_path}/"
            return list(self._bucket.list_blobs(prefix=prefix))

        try:
            blobs = await loop.run_in_executor(
                get_executors().io_executor,
                list_blobs,
            )
        except Exception as e:
            logger.error(f"Error listing folder {folder_path}: {e}")
            return None

        # Filter out system files
        doc_blobs = [
            b for b in blobs
            if not b.name.endswith("/.folder_created")
            and self._is_supported_file(b.name)
        ]

        if not blobs:  # No blobs at all means folder doesn't exist
            return None

        # Calculate totals
        total_size = sum(b.size or 0 for b in doc_blobs)

        return BulkFolderInfo(
            folder_name=folder_name,
            gcs_path=self._get_full_gcs_uri(folder_path),
            document_count=len(doc_blobs),
            total_size_bytes=total_size,
            created_at=datetime.utcnow(),  # Would need metadata for actual time
            org_id=org_id,
        )

    async def list_folders(self, org_id: str) -> List[BulkFolderInfo]:
        """
        List all bulk folders for an organization.

        Args:
            org_id: Organization ID

        Returns:
            List of folder info objects
        """
        prefix = f"{org_id}/{self.config.bulk_folder_prefix}/"

        loop = asyncio.get_running_loop()

        def list_prefixes():
            # Use delimiter to get "folders"
            iterator = self._bucket.list_blobs(prefix=prefix, delimiter="/")
            # Need to consume iterator to get prefixes
            list(iterator)
            return list(iterator.prefixes)

        try:
            prefixes = await loop.run_in_executor(
                get_executors().io_executor,
                list_prefixes,
            )
        except Exception as e:
            logger.error(f"Error listing folders for {org_id}: {e}")
            return []

        folders = []
        for prefix_path in prefixes:
            # Extract folder name from prefix
            # prefix_path format: org_id/bulk/folder_name/
            parts = prefix_path.rstrip("/").split("/")
            if len(parts) >= 3:
                folder_name = parts[-1]
                folder_info = await self.get_folder_info(org_id, folder_name)
                if folder_info:
                    folders.append(folder_info)

        return folders

    async def list_documents(
        self,
        org_id: str,
        folder_name: str,
    ) -> List[str]:
        """
        List all documents in a bulk folder.

        Args:
            org_id: Organization ID
            folder_name: Name of the folder

        Returns:
            List of full GCS URIs for documents
        """
        folder_path = self._get_bulk_path(org_id, folder_name)
        prefix = f"{folder_path}/"

        loop = asyncio.get_running_loop()

        def list_blobs():
            return list(self._bucket.list_blobs(prefix=prefix))

        try:
            blobs = await loop.run_in_executor(
                get_executors().io_executor,
                list_blobs,
            )
        except Exception as e:
            logger.error(f"Error listing documents in {folder_path}: {e}")
            return []

        # Filter to supported file types
        documents = [
            self._get_full_gcs_uri(b.name)
            for b in blobs
            if self._is_supported_file(b.name)
        ]

        return documents

    async def generate_upload_urls(
        self,
        org_id: str,
        folder_name: str,
        filenames: List[str],
        expiration_minutes: Optional[int] = None,
    ) -> List[SignedUrlInfo]:
        """
        Generate signed URLs for direct GCS upload.

        Args:
            org_id: Organization ID
            folder_name: Target folder name
            filenames: List of filenames to generate URLs for
            expiration_minutes: URL expiration time (uses config default if not specified)

        Returns:
            List of signed URL info objects
        """
        if expiration_minutes is None:
            expiration_minutes = self.config.signed_url_expiration_minutes

        folder_path = self._get_bulk_path(org_id, folder_name)
        expiration = timedelta(minutes=expiration_minutes)
        expires_at = datetime.utcnow() + expiration

        loop = asyncio.get_running_loop()
        signed_urls = []

        for filename in filenames:
            # Validate file extension
            if not self._is_supported_file(filename):
                logger.warning(f"Skipping unsupported file: {filename}")
                continue

            blob_path = f"{folder_path}/{filename}"
            blob = self._bucket.blob(blob_path)

            # Determine content type
            content_type = self._get_content_type(filename)

            def generate_url(b, ct):
                return b.generate_signed_url(
                    version="v4",
                    expiration=expiration,
                    method="PUT",
                    content_type=ct,
                )

            try:
                signed_url = await loop.run_in_executor(
                    get_executors().io_executor,
                    partial(generate_url, blob, content_type),
                )

                signed_urls.append(SignedUrlInfo(
                    filename=filename,
                    signed_url=signed_url,
                    gcs_path=self._get_full_gcs_uri(blob_path),
                    expires_at=expires_at,
                    content_type=content_type,
                ))

            except Exception as e:
                logger.error(f"Failed to generate signed URL for {filename}: {e}")
                continue

        logger.info(f"Generated {len(signed_urls)} signed URLs for folder {folder_name}")
        return signed_urls

    async def validate_folder_limit(
        self,
        org_id: str,
        folder_name: str,
    ) -> tuple[bool, str]:
        """
        Validate folder meets processing requirements.

        Args:
            org_id: Organization ID
            folder_name: Folder name

        Returns:
            Tuple of (is_valid, message)
        """
        info = await self.get_folder_info(org_id, folder_name)

        if not info:
            return False, "Folder not found or empty"

        if info.document_count == 0:
            return False, "Folder contains no supported documents"

        if info.document_count > self.config.max_documents_per_folder:
            return False, (
                f"Folder has {info.document_count} documents, "
                f"max is {self.config.max_documents_per_folder}"
            )

        return True, "Folder is valid for processing"

    async def folder_exists(self, org_id: str, folder_name: str) -> bool:
        """
        Check if a bulk folder exists.

        Args:
            org_id: Organization ID
            folder_name: Folder name

        Returns:
            True if folder exists
        """
        folder_path = self._get_bulk_path(org_id, folder_name)
        placeholder_path = f"{folder_path}/.folder_created"

        blob = self._bucket.blob(placeholder_path)

        loop = asyncio.get_running_loop()
        try:
            exists = await loop.run_in_executor(
                get_executors().io_executor,
                blob.exists,
            )
            return exists
        except Exception:
            return False

    def _is_supported_file(self, filename: str) -> bool:
        """Check if file has a supported extension."""
        path = Path(filename)
        suffix = path.suffix.lower()

        # Skip hidden files and system files
        if path.name.startswith("."):
            return False

        return suffix in self.config.supported_extensions

    def _get_content_type(self, filename: str) -> str:
        """Get content type for a filename."""
        suffix = Path(filename).suffix.lower()

        content_types = {
            ".pdf": "application/pdf",
            ".doc": "application/msword",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".ppt": "application/vnd.ms-powerpoint",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".xls": "application/vnd.ms-excel",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".csv": "text/csv",
            ".txt": "text/plain",
            ".rtf": "application/rtf",
            ".html": "text/html",
            ".htm": "text/html",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".tiff": "image/tiff",
            ".webp": "image/webp",
        }

        return content_types.get(suffix, "application/octet-stream")


# Singleton instance
_folder_manager: Optional[FolderManager] = None


def get_folder_manager() -> FolderManager:
    """Get the folder manager singleton."""
    global _folder_manager
    if _folder_manager is None:
        _folder_manager = FolderManager()
    return _folder_manager


def reset_folder_manager() -> None:
    """Reset the folder manager singleton (for testing)."""
    global _folder_manager
    _folder_manager = None
