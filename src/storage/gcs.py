"""Google Cloud Storage backend implementation."""

import asyncio
import logging
import threading
from functools import partial
from typing import Optional, List

from google.cloud import storage

from src.core.executors import get_executors
from google.cloud.exceptions import NotFound, Forbidden
from google.api_core.exceptions import GoogleAPIError

from .base import StorageBackend
from src.utils.gcs_utils import is_gcs_path, parse_gcs_uri

logger = logging.getLogger(__name__)

# Module-level singleton for GCS client (thread-safe)
_gcs_client: Optional[storage.Client] = None
_gcs_client_lock = threading.Lock()


def _get_gcs_client() -> storage.Client:
    """Get or create the singleton GCS client (thread-safe)."""
    global _gcs_client
    if _gcs_client is None:
        with _gcs_client_lock:
            if _gcs_client is None:
                _gcs_client = storage.Client()
                logger.info("GCS client initialized (singleton)")
    return _gcs_client


class GCSStorage(StorageBackend):
    """Google Cloud Storage implementation using Application Default Credentials."""

    def __init__(self, bucket_name: str, prefix: str = ""):
        """
        Initialize GCS storage.

        Args:
            bucket_name: GCS bucket name
            prefix: Prefix/folder within bucket (e.g., "demo_docs")
        """
        self.bucket_name = bucket_name
        self.prefix = prefix.strip("/")

        # Use singleton client for efficiency
        self._client = _get_gcs_client()
        self._bucket = self._client.bucket(bucket_name)

        logger.info(f"GCS storage initialized: gs://{bucket_name}/{prefix}")

    def _get_blob_name(self, filename: str, directory: str = "") -> str:
        """Construct full blob name with prefix and directory."""
        parts = [p for p in [self.prefix, directory, filename] if p]
        return "/".join(parts)

    async def save(self, content: str, filename: str, directory: str = "", use_prefix: bool = True) -> str:
        """Save content to GCS.

        Args:
            content: Content to save
            filename: Name of the file
            directory: Directory path within storage
            use_prefix: If True, prepend the configured GCS prefix. If False, use directory as-is.
        """
        if use_prefix:
            blob_name = self._get_blob_name(filename, directory)
        else:
            # Skip prefix - use directory and filename directly
            parts = [p for p in [directory, filename] if p]
            blob_name = "/".join(parts)
        blob = self._bucket.blob(blob_name)

        # Determine content type
        content_type = "text/markdown" if filename.endswith(".md") else "application/json"

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                get_executors().io_executor,
                partial(blob.upload_from_string, content, content_type=content_type),
            )
        except GoogleAPIError as e:
            logger.error(f"Failed to save to GCS: {e}")
            raise

        uri = f"gs://{self.bucket_name}/{blob_name}"
        logger.info(f"Saved to GCS: {uri}")
        return uri

    async def save_bytes(
        self,
        content: bytes,
        filename: str,
        directory: str = "",
        use_prefix: bool = True,
    ) -> str:
        """Save binary content to GCS.

        Args:
            content: Binary content to save
            filename: Name of the file
            directory: Directory path within storage
            use_prefix: If True, prepend the configured GCS prefix. If False, use directory as-is.

        Returns:
            Full GCS URI (gs://bucket/path)
        """
        if use_prefix:
            blob_name = self._get_blob_name(filename, directory)
        else:
            # Skip prefix - use directory and filename directly
            parts = [p for p in [directory, filename] if p]
            blob_name = "/".join(parts)

        blob = self._bucket.blob(blob_name)

        # Determine content type from extension
        content_type = self._get_content_type_for_extension(filename)

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                get_executors().io_executor,
                partial(blob.upload_from_string, content, content_type=content_type),
            )
        except GoogleAPIError as e:
            logger.error(f"Failed to save bytes to GCS: {e}")
            raise

        uri = f"gs://{self.bucket_name}/{blob_name}"
        logger.info(f"Saved bytes to GCS: {uri} ({len(content)} bytes)")
        return uri

    def _get_content_type_for_extension(self, filename: str) -> str:
        """Get content type based on file extension."""
        from pathlib import Path

        ext = Path(filename).suffix.lower()
        content_types = {
            ".pdf": "application/pdf",
            ".doc": "application/msword",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".ppt": "application/vnd.ms-powerpoint",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".xls": "application/vnd.ms-excel",
            ".csv": "text/csv",
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".tiff": "image/tiff",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
            ".html": "text/html",
            ".htm": "text/html",
            ".json": "application/json",
            ".rtf": "application/rtf",
        }
        return content_types.get(ext, "application/octet-stream")

    async def read(self, path: str, use_prefix: bool = True) -> Optional[str]:
        """Read content from GCS.

        Args:
            path: GCS URI (gs://bucket/path) or relative path
            use_prefix: If True, prepend configured prefix to relative paths.
                       If False, use path directly without prefix.
        """
        # Handle gs:// URIs and relative paths
        if is_gcs_path(path):
            _, blob_name = parse_gcs_uri(path)
            if not blob_name:
                return None
        elif use_prefix:
            blob_name = self._get_blob_name(path)
        else:
            # Use path directly without prefix
            blob_name = path

        blob = self._bucket.blob(blob_name)

        loop = asyncio.get_running_loop()
        try:
            content = await loop.run_in_executor(get_executors().io_executor, blob.download_as_text)
            return content
        except NotFound:
            logger.debug(f"Blob not found: {blob_name}")
            return None
        except Forbidden as e:
            logger.error(f"Permission denied reading {blob_name}: {e}")
            raise
        except GoogleAPIError as e:
            logger.error(f"GCS error reading {blob_name}: {e}")
            raise

    async def exists(self, path: str, use_prefix: bool = True) -> bool:
        """Check if blob exists in GCS.

        Args:
            path: GCS URI (gs://bucket/path) or relative path
            use_prefix: If True, prepend configured prefix to relative paths.
                       If False, use path directly without prefix.
        """
        if is_gcs_path(path):
            _, blob_name = parse_gcs_uri(path)
            if not blob_name:
                return False
        elif use_prefix:
            blob_name = self._get_blob_name(path)
        else:
            # Use path directly without prefix
            blob_name = path

        blob = self._bucket.blob(blob_name)

        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(get_executors().io_executor, blob.exists)
        except GoogleAPIError as e:
            logger.error(f"GCS error checking existence of {blob_name}: {e}")
            return False

    async def list_files(
        self, directory: str, extension: Optional[str] = None, use_prefix: bool = True
    ) -> List[str]:
        """List blobs in a directory prefix.

        Args:
            directory: Directory path to list
            extension: Filter by file extension (e.g., ".json")
            use_prefix: If True, prepend the configured GCS prefix. If False, use directory as-is.
        """
        if use_prefix:
            prefix = self._get_blob_name("", directory)
        else:
            prefix = directory

        if prefix and not prefix.endswith("/"):
            prefix += "/"

        loop = asyncio.get_running_loop()
        try:
            blobs = await loop.run_in_executor(
                get_executors().io_executor,
                partial(list, self._client.list_blobs(self.bucket_name, prefix=prefix)),
            )
        except GoogleAPIError as e:
            logger.error(f"GCS error listing {prefix}: {e}")
            return []

        files = []
        for blob in blobs:
            if extension is None or blob.name.endswith(extension):
                files.append(f"gs://{self.bucket_name}/{blob.name}")

        return files

    async def download_bytes(self, path: str) -> Optional[bytes]:
        """Download file content as bytes from GCS.

        Used for binary files (PDF, images, etc.) that need to be
        downloaded for local processing.

        Args:
            path: GCS URI (gs://bucket/path) or relative path

        Returns:
            File content as bytes, or None if not found
        """
        # Handle gs:// URIs and relative paths
        if is_gcs_path(path):
            _, blob_name = parse_gcs_uri(path)
            if not blob_name:
                return None
        else:
            blob_name = self._get_blob_name(path)

        blob = self._bucket.blob(blob_name)

        loop = asyncio.get_running_loop()
        try:
            content = await loop.run_in_executor(get_executors().io_executor, blob.download_as_bytes)
            logger.debug(f"Downloaded {len(content)} bytes from GCS: {path}")
            return content
        except NotFound:
            logger.debug(f"Blob not found: {blob_name}")
            return None
        except Forbidden as e:
            logger.error(f"Permission denied reading {blob_name}: {e}")
            raise
        except GoogleAPIError as e:
            logger.error(f"GCS error reading {blob_name}: {e}")
            raise

    async def delete(self, path: str) -> bool:
        """Delete blob from GCS."""
        if is_gcs_path(path):
            _, blob_name = parse_gcs_uri(path)
            if not blob_name:
                return False
        else:
            blob_name = self._get_blob_name(path)

        blob = self._bucket.blob(blob_name)

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(get_executors().io_executor, blob.delete)
            logger.info(f"Deleted from GCS: gs://{self.bucket_name}/{blob_name}")
            return True
        except NotFound:
            return False
        except GoogleAPIError as e:
            logger.error(f"GCS error deleting {blob_name}: {e}")
            raise

    def get_uri(self, path: str) -> str:
        """Get GCS URI for a path."""
        if is_gcs_path(path):
            return path
        blob_name = self._get_blob_name(path)
        return f"gs://{self.bucket_name}/{blob_name}"
